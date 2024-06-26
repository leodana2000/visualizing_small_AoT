import torch as t
import numpy as np
from typing import List, Tuple
from models import Transformer
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from utils import layer_norm, generate_each, generate_uniform_sphere, generate_uniform_simplex
import plotly.graph_objects as go #type: ignore
import plotly.express as px #type: ignore
import ternary #type: ignore


def colored(ind: int) -> Tuple[str, str]:
    """Define a color and colorscale per class."""
    assert ind<5, "Coloscale only available for 5 classes."
    if ind == 0:
        c = 'Greens'
        c2 = 'green'
    elif ind == 1:
        c = 'Reds'
        c2 = 'red'
    elif ind == 2:
        c = 'Blues'
        c2 = 'blue'
    elif ind == 3:
        c = 'Purples'
        c2 = 'purple'
    elif ind == 4:
        c = 'Oranges'
        c2 = 'orange'

    return (c, c2)


def color_classes(softmax_output: t.Tensor, nb_points: int, nb_classes: int):
    """
    For a set of prediction, find each prediction's class and color.
    """

    norm = Normalize(vmin=0, vmax=1)
    combined_colors = np.zeros((nb_points, nb_points, 4))
    max_proba = t.max(softmax_output, dim=-1)[0]

    for out_token in range(nb_classes):
        # For each classes, find the points that were predicted to be in this class
        proba = softmax_output[:, :, out_token].detach()
        mask = (proba == max_proba)

        # Give the point the class' color
        cmap = plt.get_cmap(colored(out_token)[0])
        color = cmap(norm(proba))
        combined_colors[mask] = color[mask]
    
    return combined_colors


def torus_grid(wind_down: float=-np.pi, wind_up: float=np.pi, nb_points: int=2000) -> t.Tensor:
    """
    Generates nb_points uniformly on the torus.
    """

    # Generate a grid on the square [0,1]^2
    theta = np.linspace(wind_down, wind_up, nb_points)
    phi = np.linspace(wind_down, wind_up, nb_points)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.flatten(), phi.flatten()

    # Transform the square into a Torus
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = np.cos(phi)
    y2 = np.sin(phi)

    # Store the grid in a tensor
    data = np.stack([np.stack([x1, y1, np.zeros_like(x1)], axis=-1), 
                    np.stack([x2, y2, np.zeros_like(x2)], axis=-1)], axis=1)
    data = t.tensor(data, dtype=t.float32)  

    return data


def compute_attention_pattern(model: Transformer, module: int, head: int, 
                              layer: int, data: t.Tensor, hardmax: bool
                              ) -> t.Tensor:
    """
    Computes the attention pattern of the right module, head and layer.
    Use the 'hardmax' parameter to binarize the softmax. 
    'data' should be the output of the function 'torus_grid'.
    """
    nb_points = int(np.sqrt(data.shape[0]))

    data_normed = layer_norm(data)
    data_positionned = data_normed + model.pos_emb.weight.detach()[:2].unsqueeze(0)

    attn_mask = t.tril(t.ones((2, 2))) == 0
    key_position = 0
    query_position = 1

    attn = model.attn_seq[layer][module]
    _, attention_pattern = attn(data_positionned, data_positionned, data_positionned, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
    attention_pattern = attention_pattern[:, head, query_position, key_position].reshape((nb_points, nb_points)).detach()
    
    if hardmax:
        attention_pattern = (attention_pattern > 0.5).to(t.int)

    return attention_pattern


def get_residual_stream(model: Transformer, nb_points: int=2000, 
                        wind_down: float=-np.pi, wind_up: float=np.pi
                        ) -> Tuple[t.Tensor, str]:
    """
    Gives the result of the RGB contribution corresponding to the residual stream.
    """
    assert model.meta_params['d'] == 3, "Your model needs to have embedding space dimension 3."

    data = torus_grid(nb_points=nb_points, wind_down=wind_down, wind_up=wind_up)[:, 1].reshape((nb_points, nb_points, 3))
    normed_data = layer_norm(data)
    positioned_data = normed_data + model.pos_emb.weight[1].unsqueeze(0).unsqueeze(0)
    
    return (positioned_data, "Residual stream of the second token.")


def token_mixing_torus(model: Transformer, module: int, head: int,layer: int,
                       mode: str="both", use_OV: bool=True,
                       nb_points: int=2000, hardmax=False, 
                       wind_down: float=-np.pi, wind_up: float=np.pi
                       ) -> Tuple[t.Tensor, str]:
    """
    Plots the embedding space after the attention mechanisme on a Torus.
    If 'use_OV', the value-output matrix multiplication is applied, 
    otherwise this is just token mixing.

    The 'mode' parameter encodes for the ways to mix token pairs:
    * mode == 'both' for the same computation as in the attention head,
    * mode == 'pos' to only mix the positional embeddings ,
    * mode == 'word' to only mix the word embeddings,

    Works only for embedding dimension 3, and 3 tokens.
    """
    assert model.meta_params['d'] == 3, f"Your transformer needs to have embedding dimenion of 3"
    assert model.meta_params['context_window'] == 3, f"Your Transformer needs to take exactly 2 tokens"
    
    # Generate points on the Torus
    data = torus_grid(wind_down=wind_down, wind_up=wind_up, nb_points=nb_points)

    # Compute the attention pattern
    attention_pattern = compute_attention_pattern(model, module, head, layer, data, hardmax).unsqueeze(-1)

    # Choose which mode to use for the token mixing
    data_normed = layer_norm(data)
    if mode == "both":
        data_to_mix = data_normed + model.pos_emb.weight.detach()[:2].unsqueeze(0)
        title = "Word and position embedding mix on a Torus"
    elif mode == "pos":
        data_to_mix = t.zeros_like(data_normed) + model.pos_emb.weight.detach()[:2].unsqueeze(0)
        title = "Position embedding mix on a Torus"
    elif mode == "word":
        data_to_mix = data_normed
        title = "Word embedding mix on a Torus"
    data_to_mix = data_to_mix.reshape((nb_points, nb_points, 2, 3))

    # Mix both token with the attention pattern
    mixed_data = data_to_mix[:, :, 0]*attention_pattern + data_to_mix[:, :, 1]*(1-attention_pattern)

    if use_OV:
        # Reproduce the W_{OV} multiplication of an attention head
        nb_head = model.meta_params['nb_head']
        d = 3
        W_O = model.attn_seq[layer][module].out_proj.weight.mH.split([d//nb_head]*nb_head)[head].detach()
        W_V = model.attn_seq[layer][module].v_proj_weight.split([d//nb_head]*nb_head)[head].detach()
        mixed_data = t.einsum('Nnd, Ad, AD -> NnD', mixed_data, W_V, W_O)
        title += " after output matrix"
    else:
        title += " before output matrix"

    if hardmax:
        title += ", hardmaxed"
    title += f"\n for layer {layer}, module {module} and head {head}"

    return (mixed_data, title)


def plot_unemb(model: Transformer, nb_points: int=10000) -> None:
    """
    Plots the unembedding's clustering on the unit sphere in 3d.
    """
    assert model.meta_params['d'] == 3, "Your model embedding dimension needs to be 3."

    # Sample uniformly on the sphere and plot each class
    sphere_coordinate = generate_uniform_sphere(nb_points)
    color_list = px.colors.qualitative.Plotly
    compatible_colors = []

    W_U = model.unemb.weight.detach()
    for x,y,z in zip(*sphere_coordinate):
        unit_vect = t.Tensor([x,y,z])
        compatible_colors.append(color_list[t.argmax(W_U@unit_vect, dim=0).item()])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sphere_coordinate[0], y=sphere_coordinate[1], z=sphere_coordinate[2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=compatible_colors,
                    opacity=1,
                    symbol='cross'
    ))])

    # Adding special points representing the pair-input vector, as well as the columns of W_U
    special_points: List[List[t.Tensor]] = [[], [], []]
    special_colors = []

    # We create each possible token pair in a bacthed tensor
    dict_size = model.meta_params['N']
    examples = generate_each(model.pi, eps=0.1/(dict_size**2))
    indices = t.zeros(examples.shape[0])
    for i, ex in enumerate(examples):
        indices[i] = ex[0]+ex[1]*dict_size
    indices = indices.to(t.int)

    # We retreive the computations of each token pair after the attention layer
    with t.no_grad():
        _, computations = model.forward(examples, out_computation=True)
    att_output = computations[f'res_after_mlp_layer_{0}'].detach()[:, 1, :]

    # Add the pair input token (balck points)
    att_output = att_output/t.norm(att_output, dim=-1, keepdim=True)
    for x,y,z in zip(*att_output.mH):
        special_points[0].append(x)
        special_points[1].append(y)
        special_points[2].append(z)
        special_colors.append('black')

    # Add the centers of the Voronoi cells (white points)
    W_U = W_U/t.norm(W_U, dim=-1, keepdim=True)
    for x,y,z in zip(*W_U.mH):
        special_points[0].append(x)
        special_points[1].append(y)
        special_points[2].append(z)
        special_colors.append('white')

    fig.add_trace(go.Scatter3d(
        x=special_points[0],
        y=special_points[1],
        z=special_points[2],
        mode='markers',
        marker=dict(
            size=10,
            color=special_colors,
            opacity=1
        )
    ))
    fig.update_layout(
        title="Clustering algorithm of the Unembedding",
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            aspectmode='data'
        )
    )
    fig.show()


def plot_attention_torus(model: Transformer, module: int, head: int, 
                    layer: int = 0, nb_points: int=2000, hardmax=False, 
                    wind_down: float=-np.pi, wind_up: float=np.pi) -> None:
    """
    Plots the probability torus of the tokens at position 0, for a given token at position 1. 
    Works only for embedding dimension 3.
    """
    assert model.meta_params['d'] == 3
    assert model.meta_params['context_window'] == 3

    # Generate points on the Torus
    data = torus_grid(wind_down=wind_down, wind_up=wind_up, nb_points=nb_points)

    # Compute the attention pattern
    attention_pattern = compute_attention_pattern(model, module, head, layer, data, hardmax)

    # Prepare for plotting
    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.get_cmap('Greys')
    norm = Normalize(vmin=0, vmax=1)

    # Plotting with imshow
    im = ax.imshow(attention_pattern, cmap=cmap, norm=norm, extent=(wind_down, wind_up, wind_down, wind_up), origin='lower', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Probability')

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Probability that the second token attend to the first on a Torus\n module={module}, head={head}, layer={layer}')
    plt.xticks([0, np.pi/2, np.pi, -np.pi/2, -np.pi], ['0', 'π/2', 'π', '-π/2', '-π'])
    plt.yticks([0, np.pi/2, np.pi, -np.pi/2, -np.pi], ['0', 'π/2', 'π', '-π/2', '-π'])
    plt.show()


def plot_RGB_torus(unscaled_RGBs: List[Tuple[t.Tensor, str]],
                   wind_down: float=-np.pi, wind_up: float=np.pi) -> None:
    """
    If you wish to visualize a head, pass down the list of visualization that you want,
    and the algorithm will find the clipping parameters that make the plot summable,
    meaning that in the RGB space: RGB(im_1)+RGB(im_2)=RGB(im_1+im_2).

    Only put in the list comparable RGBs.
    """

    # Compute the minimum and maximum values of the embedding space
    Mini = t.empty(3)
    Maxi = t.empty(3)
    for (mixed_data, _) in unscaled_RGBs:
        Mini = t.min(Mini, t.min(mixed_data.flatten(0, 1), dim=0)[0])
        Maxi = t.max(Maxi, t.max(mixed_data.flatten(0, 1), dim=0)[0])

    # Transform the vectors into colored pixels and plot them
    for (mixed_data, title) in unscaled_RGBs:
        RGB = (mixed_data-Mini)/(Maxi-Mini)

        # Plot the mix
        _, ax = plt.subplots(1, 1, figsize=(10, 8))
        _ = ax.imshow(RGB.detach(), extent=(wind_down, wind_up, wind_down, wind_up), origin='lower', aspect='auto')
        ax.set_xlabel('Angle Position 1')
        ax.set_ylabel('Angle Position 2')
        ax.set_title(title)
        plt.xticks([0, np.pi/2, np.pi, -np.pi/2, -np.pi], ['0', 'π/2', 'π', '-π/2', '-π'])
        plt.yticks([0, np.pi/2, np.pi, -np.pi/2, -np.pi], ['0', 'π/2', 'π', '-π/2', '-π'])
        plt.show()


def plot_classes_torus(unscaled_RGBs: List[Tuple[t.Tensor, str]], model: Transformer,
                       wind_down: float=-np.pi, wind_up: float=np.pi) -> None:
    """
    For a list of unscaled RGBs, combine them and plot their clustering map on the torus.
    This function can be used to plot the clustering of a single head.
    """
    assert unscaled_RGBs != [], "Your need a non-empty set of RGB to combine."
    nb_points = unscaled_RGBs[0][0].shape[0]

    # Combine the output heads
    embedding_output = t.zeros((nb_points, nb_points, 3))
    for (RGB, _) in unscaled_RGBs:
        embedding_output += RGB
    unembedding_output = model.unemb(embedding_output)

    # Transform into probabilities
    softmax_output = t.softmax(unembedding_output, dim=-1)

    # Find the color for each point
    combined_colors = color_classes(softmax_output, nb_points, model.meta_params['N'])

    # Plot the image
    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(combined_colors, extent=(wind_down, wind_up, wind_down, wind_up), origin='lower', aspect='auto')

    # Plot the token pairs on the torus
    W_E = layer_norm(model.word_emb.weight.detach())
    for i, v0 in enumerate(W_E):
        for j, v1 in enumerate(W_E):
            # Convert the vectors back to angles
            angle_x = t.atan2(v0[1]-v0[2], v0[0]-v0[2])
            angle_y = t.atan2(v1[1]-v1[2], v1[0]-v1[2])

            # Find the color corresponding to the class of the input
            next_token = int(t.argmax(model.pi[2][i, j]).item())
            color_xy = colored(next_token)[1]

            ax.scatter(angle_x, angle_y, color=color_xy, edgecolor='black')

    ax.set_xlabel('Angle Position 1')
    ax.set_ylabel('Angle Position 2')
    ax.set_title('Output probability of the most likely class on a Torus')
    plt.xticks([0, np.pi/2, np.pi, -np.pi/2, -np.pi], ['0', 'π/2', 'π', '-π/2', '-π'])
    plt.yticks([0, np.pi/2, np.pi, -np.pi/2, -np.pi], ['0', 'π/2', 'π', '-π/2', '-π'])
    plt.show()


def plot_classes_full_torus(model: Transformer, wind_down: float=-np.pi, wind_up: float=np.pi, nb_points: int=2000) -> None:
    """
    This function is a short cut to visualize the classes of the Transformer on the torus.
    If you wan to visualize only a subset of the heads, please use 'plot_classes_torus'.
    """
    assert model.meta_params['nb_layers'] == 1, "Please do not use this function with more than 1 layer."
    assert model.meta_params['d'] == 3, "Your Transformer needs to have an embedding dimension of 3."
    assert model.meta_params['context_window'] == 3, "Your Transformer needs to be taking only 2 input tokens."
    
    unscaled_RGBs = []
    layer = 0
    nb_att_mod = model.meta_params['para']
    nb_head = model.meta_params['nb_head']

    unscaled_RGBs.append(get_residual_stream(model, wind_down=wind_down, wind_up=wind_up, nb_points=nb_points))
    for module in range(nb_att_mod):
        for head in range(nb_head):
            unscaled_RGBs.append(
                token_mixing_torus(
                    model,
                    module,
                    head,
                    layer,
                    wind_down=wind_down,
                    wind_up=wind_up,
                    nb_points=nb_points
                )
            )
        
    plot_classes_torus(
        unscaled_RGBs,
        model,
        wind_down=wind_down,
        wind_up=wind_up
    )


def mapping_torus(model: Transformer, normed_shape: bool=False, 
                  nb_points: int=100, wind_down: float=-np.pi, wind_up: float=np.pi) -> None:
    """
    Represents in 3d the output of the attention, whcih is continuously deformed torus. 
    It can be normalized for better visibility using 'normed_shape'.

    Be carefull to the number of points, it could make plotly crash.
    """
    assert model.meta_params['d'] == 3, f"Your transformer needs to have embedding dimenion of 3"
    assert model.meta_params['context_window'] == 3, f"Your Transformer needs to take exactly 2 tokens"
    
    # Generate points on the Torus
    data = torus_grid(wind_down=wind_down, wind_up=wind_up, nb_points=nb_points)

    # Layer normalization
    data_normed = layer_norm(data)

    # Throught the model
    _, computation = model.forward(data_normed, continuous=True, out_computation=True)
    d = model.meta_params['d']
    layer = 0
    final_embedding = computation[f'res_after_attn_layer_{layer}'].detach()[:, 1].reshape((nb_points, nb_points, d))

    # Normalize the shape in 3d for better visibility
    text = ""
    if normed_shape:
        U, _, V = t.linalg.svd(final_embedding.flatten(0, 1), full_matrices=False)
        final_embedding = (U@V).reshape((nb_points, nb_points, d))
        text += ", normalied"

    # Apply the function
    x, y, z = final_embedding[:, :, 0], final_embedding[:, :, 1], final_embedding[:, :, 2]

    # Create the angles for the color scale
    theta = np.linspace(wind_down, wind_up, nb_points)
    phi = np.linspace(wind_down, wind_up, nb_points)
    theta, phi = np.meshgrid(theta, phi)

    maxi = t.max(final_embedding)
    mini = t.min(final_embedding)
    Dict = dict(nticks=4, range=[mini, maxi])

    # Create the 3D plot for the first angle
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z, 
            surfacecolor=theta, 
            colorscale='Reds', 
            showscale=False, 
            opacity=0.5
        )
    )
    fig.update_layout(
        title='Image of the torus throught the attention, color representing first angle'+text, 
        scene=dict(
            xaxis=Dict,
            yaxis=Dict,
            zaxis=Dict
        )
    )
    fig.show()

    # Create the 3D plot for the first angle
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z, 
            surfacecolor=phi, 
            colorscale='Blues', 
            showscale=False, 
            opacity=0.5
        )
    )
    fig.update_layout(
        title='Image of the torus throught the attention, color representing second angle'+text, 
        scene=dict(
            xaxis=Dict,
            yaxis=Dict,
            zaxis=Dict
        )
    )
    fig.show()
    

def back_track(model: Transformer, examples: t.Tensor, seq_target: int = 2) -> List[List[List[np.ndarray]]]:
    """
    Computes the contribution of each head to each logit.
    The contribution of the layer l, parallel module p in the direction id is:
        contribution_list[id][l][p]
    and for the residual stream's contribution, it is:
        contribution_list[id][-1][0]
    """

    with t.no_grad():
        _, computation = model.forward(examples, out_computation=True)

        N = model.meta_params['N']
        nb_layers = model.meta_params['nb_layers']
        paras = model.meta_params['para']

        contribution_list = []
        for id in range(N):
            direction = model.unemb.weight[id]

            id_list = []
            for layer in range(nb_layers):
                layer_list = []
                for para in range(paras):
                    layer_list.append(np.array(t.einsum('d, ...d -> ...', direction, computation[f'para_{para}_layer_{layer}'][:, seq_target-1]).detach()))
                id_list.append(layer_list)

                id_list.append([np.array(t.einsum('d, ...d -> ...', direction, computation[f'res_{layer}'][:, seq_target-1]).detach())])
                id_list.append([np.array(t.einsum('d, ...d -> ...', direction, computation[f'res_after_attn_layer_{layer}'][:, seq_target-1]).detach())])
            contribution_list.append(id_list)

    return contribution_list #Shape: [N, nb_layers, nb_para, batch_size]


def plot_accuracy_simplex(
        model: Transformer,
        seq_target: int = 2,
        comp_method: str = 'combined',
        input=None, 
        output=None, 
        nb_points: int=5000,
    ) -> None:
    """
    Plots the accuracy of a positive combination of heads on a simplex (when they are 3).
    """
    assert model.meta_params['nb_layers'] == 1
    assert comp_method in ['combined', 'input', 'output']
    if comp_method == 'input':
        assert input is not None
    elif comp_method == 'output':
        assert output is not None

    N = model.meta_params['N']
    layer = 0

    examples = generate_each(model.pi)
    contribution = back_track(model, examples)
    contrib = t.tensor(np.array([contrib[:-2] for contrib in contribution]))


    # Computes the contribution according to the good method.
    if comp_method == 'input':
        select = examples[input, seq_target].to(t.int)
    elif comp_method == 'output':
        select = (examples[:, seq_target] == output)
    elif  comp_method == 'combined':
        select = t.arange(len(examples))

    next_tokens = examples[:, seq_target].to(t.int)
    nb_tokens = len(next_tokens)
    contrib_l_para = contrib[:, layer, :, :]
    contrib_max = contrib_l_para[next_tokens, :, t.arange(nb_tokens)].mH.unsqueeze(0)
    contrib = contrib_l_para-contrib_max # This operation means we cancel the residual stream ! 

    # Sample data points (p, q, r) on the simplex
    simplex_coordinate = generate_uniform_simplex(nb_points=nb_points, add_special=True)
    simplex_value = []
    for p, q, r in simplex_coordinate:
        mixture = contrib[:, 0]*p + contrib[:, 1]*q + contrib[:, 2]*r

        acc = (t.Tensor([t.max(mixture[t.arange(N) != next_tokens[j], j], dim=0)[0].item() for j in t.arange(nb_tokens)]) <= 0).to(t.float)[select].mean().item()
        simplex_value.append(acc)

    # Initialize Colormaps and compute each points' color.
    c = 'Oranges'
    c_label = 'Accuracy'
    norm = Normalize(vmin=min(simplex_value), vmax=max(simplex_value))
    cmap = plt.get_cmap(c)
    colors = cmap(norm(simplex_value))

    # Initialize a ternary plot
    _, ax = plt.subplots(figsize=(10, 10))
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.)
    if comp_method == 'input':
        tax.set_title(f"Ternary Token {input}", fontsize=20)
    elif comp_method == 'output':
        tax.set_title(f"Ternary Output Token {output}", fontsize=20)
    elif comp_method == 'combined':
        tax.set_title("Ternary Combine Token", fontsize=20)

    # Set corner labels
    tax.right_corner_label("Head 0", fontsize=15)
    tax.top_corner_label("Head 1", fontsize=15)
    tax.left_corner_label("Head 2", fontsize=15)

    # Plot data points with colors
    for (point, color) in zip(simplex_coordinate, colors):
        tax.scatter([point], marker='o', color=color)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(c_label, fontsize=15)
        
    # Set ticks and gridlines
    tax.gridlines(color="black", multiple=0.1)
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, tick_formats="")
    tax.clear_matplotlib_ticks()

    plt.show()