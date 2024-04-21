import torch.nn as nn
import torch
from geometry import to_basis, from_basis
import torch.nn.functional as F


class SelfGatingBlock(nn.Module):
    def __init__(self, input_dim):
        super(SelfGatingBlock, self).__init__()
        # The gating mechanism is an affine transformation followed by a sigmoid activation
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply the gating mechanism
        # The gate has the same dimension as the input and applies an element-wise multiplication
        gate = self.gate(x)
        return x * gate


class SoftClusteringBlock(nn.Module):
    def __init__(self, num_features, num_clusters):
        super(SoftClusteringBlock, self).__init__()
        self.num_clusters = num_clusters
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, num_features))

    def forward(self, x):
        centers_expanded = self.cluster_centers.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, num_clusters, k)
        distances = torch.sqrt(
            torch.sum((x.unsqueeze(2) - centers_expanded) ** 2, dim=-1))  # Shape becomes (B, N, num_clusters)
        associations = F.softmax(-distances, dim=-1)  # Shape becomes (B, N, num_clusters)
        output = torch.matmul(associations, self.cluster_centers.unsqueeze(0))
        return output

class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values
    """

    def __init__(self, C_inout, method='spectral'):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method  # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            # Diffuse
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex
            x_diffuse = from_basis(x_diffuse_spec, evecs)

        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)

            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")

        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if (self.with_gradient_rotations):
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors  # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''

    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True,
                 diffusion_method='spectral',
                 with_gradient_features=True,
                 with_gradient_rotations=True):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)

        self.MLP_C = 2 * self.C_width

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width,
                                                             with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width

        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)

    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0]  # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))

        # Diffusion block
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b, ...], x_diffuse[b, ...])
                x_gradY = torch.mm(gradY[b, ...], x_diffuse[b, ...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad)

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out


class GatedDiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True,
                 diffusion_method='spectral',
                 with_gradient_features=True,
                 with_gradient_rotations=True,
                 use_clustering = False):
        super(GatedDiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims
        self.self_gating = SelfGatingBlock(input_dim=self.C_width)
        self.clustering = SoftClusteringBlock(num_features=self.C_width, num_clusters=40)
        self.use_clustering = use_clustering
        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)

        self.MLP_C = 2 * self.C_width

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width,
                                                             with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width

        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)

    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0]  # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))

        # Diffusion block
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = []  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b, ...], x_diffuse[b, ...])
                x_gradY = torch.mm(gradY[b, ...], x_diffuse[b, ...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad)

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in
        # Apply clustering
        if self.use_clustering:
            x0_out = self.clustering(x0_out)

        x0_out = self.self_gating(x0_out)
        return x0_out


class DiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices',
                 mlp_hidden_dims=None, dropout=True,
                 with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral', gated=False, use_clustering=False):
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension
            C_out (int):                    output dimension
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.gated = gated
        self.use_clustering = use_clustering

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError(
            "invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError(
            "invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        if self.gated:
            # DiffusionNet blocks
            self.blocks = []
            for i_block in range(self.N_block):
                block = GatedDiffusionNetBlock(C_width=C_width,
                                               mlp_hidden_dims=mlp_hidden_dims,
                                               dropout=dropout,
                                               diffusion_method=diffusion_method,
                                               with_gradient_features=with_gradient_features,
                                               with_gradient_rotations=with_gradient_rotations,
                                               use_clustering=self.use_clustering)

                self.blocks.append(block)
                self.add_module("block_" + str(i_block), self.blocks[-1])
        else:
            # DiffusionNet blocks
            self.blocks = []
            for i_block in range(self.N_block):
                block = DiffusionNetBlock(C_width=C_width,
                                          mlp_hidden_dims=mlp_hidden_dims,
                                          dropout=dropout,
                                          diffusion_method=diffusion_method,
                                          with_gradient_features=with_gradient_features,
                                          with_gradient_rotations=with_gradient_rotations)

                self.blocks.append(block)
                self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(self, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """

        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in:
            raise ValueError(
                "DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in, x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False

        else:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        # Apply the first linear layer
        x = self.first_lin(x_in)

        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)

        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices':
            x_out = x

        elif self.outputs_at == 'edges':
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)

        elif self.outputs_at == 'faces':
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)

        elif self.outputs_at == 'global_mean':
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant.
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)

        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out


class FunctionalMapCorrespondenceWithDiffusionNetFeatures(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, n_feat=128, input_features="xyz", lambda_param=1e-3, gated=False, use_clustering=False):
        super().__init__()
        C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features
        self.gated = gated
        self.use_clustering = use_clustering
        self.feature_extractor = DiffusionNet(
            C_in=C_in,
            C_out=n_feat,
            C_width=128,
            N_block=4,
            dropout=True,
            gated=self.gated,
            use_clustering=self.use_clustering
        )

        self.n_fmap = 30
        self.input_features = input_features
        self.lambda_param = lambda_param

    def forward(self, shape1, shape2):
        verts1, faces1, frames1, mass1, L1, evals1, evecs1, gradX1, gradY1, hks1 = shape1
        verts2, faces2, frames2, mass2, L2, evals2, evecs2, gradX2, gradY2, hks2 = shape2

        # set features
        if self.input_features == "xyz":
            features1, features2 = verts1, verts2
        elif self.input_features == "hks":
            features1, features2 = hks1, hks2

        feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                       gradX=gradX1, gradY=gradY1, faces=faces1)
        feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                       gradX=gradX2, gradY=gradY2, faces=faces2)

        return feat1, feat2


class DiffusionnetFeatureExtractor(nn.Module):
    """Diffusionnet simply as feature extractor"""

    def __init__(self, n_feat=128, input_features="xyz", lambda_param=1e-3):
        super().__init__()

        C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features

        self.feature_extractor = DiffusionNet(
            C_in=C_in,
            C_out=n_feat,
            C_width=128,
            N_block=4,
            dropout=True,
        )

        self.input_features = input_features
        self.lambda_param = lambda_param

    def forward(self, shape):
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, hks = shape

        # set features
        if self.input_features == "xyz":
            features = verts
        elif self.input_features == "hks":
            features = hks

        feat = self.feature_extractor(features, mass, L=L, evals=evals, evecs=evecs,
                                      gradX=gradX, gradY=gradY, faces=faces)
        return feat


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.num_layers = cfg["num_layers"]
        assert self.num_layers in [1, 2, 3], 'Only one, two or three layers supported'

        if self.num_layers == 1:
            self.fc = nn.Linear(cfg["in_dim"], cfg["out_dim"])
        elif self.num_layers == 2:
            self.fc1 = nn.Linear(cfg["in_dim"], cfg["h_dim"])
            self.fc2 = nn.Linear(cfg["h_dim"], cfg["out_dim"])
            self.bn1 = nn.BatchNorm1d(cfg["h_dim"])  # Batch normalization layer
        else:
            self.fc1 = nn.Linear(cfg["in_dim"], cfg["h_dim"])
            self.fc2 = nn.Linear(cfg["h_dim"], cfg["h_dim"])
            self.fc3 = nn.Linear(cfg["h_dim"], cfg["out_dim"])
            self.bn1 = nn.BatchNorm1d(cfg["h_dim"])
            self.bn2 = nn.BatchNorm1d(cfg["h_dim"])  # Additional batch norm for the second hidden layer
        self.dropout = nn.Dropout(cfg["dropout_rate"])  # Dropout layer

    def forward(self, x):
        if self.num_layers == 1:
            h = self.fc(x)
        elif self.num_layers == 2:
            h = F.relu(self.bn1(self.fc1(x)))
            h = self.dropout(h)  # Apply dropout
            h = self.fc2(h)
        else:
            h = F.relu(self.bn1(self.fc1(x)))
            h = self.dropout(h)  # Apply dropout
            h = F.relu(self.bn2(self.fc2(h)))
            h = self.dropout(h)  # Apply dropout
            h = self.fc3(h)
        return h
