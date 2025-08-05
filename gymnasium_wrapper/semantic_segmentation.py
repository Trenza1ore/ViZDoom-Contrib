"""
Semantic Segmentation support for ViZDoom Gymnasium wrapper

This module provides functionality for semantic segmentation in ViZDoom environments,
allowing objects in the game world to be categorized and colored based on their semantic classes.
"""

from collections import defaultdict, namedtuple
from functools import partial
from typing import Callable, Literal, Optional, Union

import numpy as np
from matplotlib import cm
from matplotlib.colors import Colormap

from vizdoom import GameState


SemanticClassDef = namedtuple("SemanticClassDef", ["label", "rgb"])


# External functions
def get_semantic_mapping(
    name: str,
    mapping_type: Literal["label", "rgb"],
    cmap_override: Union[None, str, Colormap] = None,
) -> partial[np.ndarray]:
    """
    Get a semantic mapping function for the specified mapping name and type.

    Arguments:
        name (str): The name of the registered semantic mapping to retrieve.
        mapping_type (Literal["label", "rgb"]): The type of mapping to return.
                                               "label" returns a function that produces label IDs.
                                               "rgb" returns a function that produces RGB colors.
        cmap_override (Union[str, Colormap]): Override the colormap if specified, no effect on label.

    Returns:
        partial[np.ndarray]: A partial function that takes a GameState and returns a semantic segmentation array.
                            For "label" type: returns array of label IDs with shape (HEIGHT, WIDTH).
                            For "rgb" type: returns RGB array with shape (3, HEIGHT, WIDTH).

    Example:
        >>> mapping_func = get_semantic_mapping("default", "label")
        >>> label_array = mapping_func(game_state)
    """
    if mapping_type == "label":
        label_def = SEMANTIC_CLASS_MAPPINGS[name][0]
        return partial(semseg, hud_padding=[-1], label_def=label_def)
    elif mapping_type == "rgb":
        if cmap_override:
            label_def = label2rgb(SEMANTIC_CLASS_MAPPINGS[name][0], cmap_override)
        else:
            label_def = SEMANTIC_CLASS_MAPPINGS[name][1]
        return partial(semseg_rgb, hud_padding=[-1], label_def=label_def)
    raise ValueError(f"Unknown mapping type: {mapping_type}")


def register_semantic_mapping(
    name: str,
    label_def: defaultdict[str, int],
    label_cmap: Union[str, Colormap] = "turbo",
    label_rgb: Optional[defaultdict[str, np.ndarray]] = None,
    force: bool = False,
) -> SemanticClassDef:
    """
    Register a new semantic mapping for use in semantic segmentation.

    Arguments:
        name (str): The name to register the mapping under. This name can be used with get_semantic_mapping().
        label_def (defaultdict[str, int]): A defaultdict mapping category names to label IDs.
        label_cmap (Union[str, Colormap]): A matplotlib colormap or colormap name to use for RGB color generation.
                                           Only used if label_rgb is None. Default: "turbo".
        label_rgb (Optional[defaultdict[str, np.ndarray]]): A defaultdict mapping category names to RGB color arrays.
                                                            If None, will be generated from label_def and label_cmap.
                                                            Default: None.
        force (bool): Whether to force register even if there is already a mapping with same name.
                      Default: False

    Returns:
        SemanticClassDef: A named tuple containing the label and RGB definitions.

    Example:
        >>> reserved = {"Floor/Ceil": 10, "Wall": 20, "Self": 250}
        >>> label_def = construct_label_def(Weapon=50, **reserved)
        >>> mapping = register_semantic_mapping("custom", custom_labels, "plasma")
        # Now can use: get_semantic_mapping("custom", "label")
    """
    if name in SEMANTIC_CLASS_MAPPINGS and not force:
        raise ValueError(f"Name {name} already registered!")
    if label_rgb is None:
        label_rgb = label2rgb(label_def, label_cmap)
    new_entry: SemanticClassDef = SemanticClassDef(label=label_def, rgb=label_rgb)
    SEMANTIC_CLASS_MAPPINGS[name] = new_entry
    return new_entry


def construct_label_def(default_factory: Optional[Callable] = None, **kwargs):
    """
    Construct a defaultdict for label definitions with custom object mappings.

    Arguments:
        default_factory (Optional[Callable]): Function to generate default value. Default: None
        **kwargs: Keyword arguments where keys are category names and values are their label IDs.

    Returns:
        defaultdict: A defaultdict that returns 0 for unknown objects and maps known objects to their label IDs.

    Example:
        >>> reserved = {"Floor/Ceil": 10, "Wall": 20, "Self": 250}
        >>> label_def = construct_label_def(Weapon=50, **reserved)
        >>> label_def["Wall"]  # Returns 20
        >>> label_def["Unknown"]  # Returns 0 (default)
    """
    if default_factory:
        return_default = default_factory
    else:

        def return_default():
            return 0

    return defaultdict(
        return_default,
        {obj_name.casefold().strip(): obj_val for obj_name, obj_val in kwargs.items()},
    )


# Internal-ish functions
def semseg(
    state: GameState,
    hud_padding: list[int],
    label_def: Union[str, defaultdict[str, int]] = "default",
) -> np.ndarray:
    """
    Perform semantic segmentation on a game state, returning label IDs. Will

    Arguments:
        state (GameState): The ViZDoom game state containing labels_buffer and labels.
        hud_padding (list[int]): int is immutable but list is, -1: auto, 0: no hud, >0: hud starting row.
        label_def (Union[str, defaultdict[str, int]]): Either a string name of a registered mapping,
                                                       or a defaultdict mapping category names to label IDs.
                                                       Default: "default".

    Returns:
        np.ndarray: A 2D array of shape (HEIGHT, WIDTH) containing label IDs for each pixel.
                    Unknown objects get label ID 0.

    Note:
        This function handles special cases for the player object:
        - If "Self" is in the label definition, the last DoomPlayer object is mapped to "Self"
        - All other objects are mapped according to their object_name
    """
    assert state.labels_buffer is not None, "labels buffer not enabled"

    label_def_: defaultdict[str, int] = (
        SEMANTIC_CLASS_MAPPINGS[label_def].label
        if isinstance(label_def, str)
        else label_def
    )

    raw_buffer: np.ndarray = state.labels_buffer
    buffer: np.ndarray = np.empty_like(raw_buffer)
    buffer[raw_buffer == 0] = label_def_["Floor/Ceil"]
    buffer[raw_buffer == 1] = label_def_["Wall"]

    cutoff = hud_padding[0]
    if cutoff:
        if cutoff < 0:
            row_is_all_zero = (raw_buffer == 0).all(axis=1)
            valid_row_indices = np.where(~row_is_all_zero)[0]

            if len(valid_row_indices) > 0:
                cutoff = hud_padding[0] = valid_row_indices[-1] + 1
        buffer[cutoff:] = 0

    if state.labels and "Self" in label_def_:
        for obj in state.labels[:-1]:
            buffer[raw_buffer == obj.value] = label_def_[obj.object_category]
        last_obj = state.labels[-1]
        if last_obj.object_category == "Player":
            buffer[raw_buffer == last_obj.value] = label_def_["Self"]
        else:
            buffer[raw_buffer == last_obj.value] = label_def_[last_obj.object_category]
    else:
        for obj in state.labels:
            buffer[raw_buffer == obj.value] = label_def_[obj.object_category]

    return buffer


def semseg_rgb(
    state: GameState,
    hud_padding: list[int],
    label_def: Union[str, defaultdict[str, np.ndarray]] = "default",
) -> np.ndarray:
    """
    Perform semantic segmentation on a game state, returning RGB colors.

    Arguments:
        state (GameState): The ViZDoom game state containing labels_buffer and labels.
        hud_padding (list[int]): int is immutable but list is, -1: auto, 0: no hud, >0: hud starting row.
        label_def (Union[str, defaultdict[str, np.ndarray]]): Either a string name of a registered mapping,
                                                              or a defaultdict mapping category names to RGB color arrays.
                                                              Default: "default".

    Returns:
        np.ndarray: A 3D array of shape (HEIGHT, WIDTH, 3) containing RGB colors for each pixel.
                    Unknown objects get the default color (corresponding to label ID 0).

    Note:
        This function handles special cases for the player object:
        - If "Self" is in the label definition, the last DoomPlayer object is mapped to "Self"
        - All other objects are mapped according to their object_name
        - The output format is (HEIGHT, WIDTH, 3) for direct use in RGB image rendering
    """
    assert state.labels_buffer is not None, "labels buffer not enabled"

    label_def_: defaultdict[str, np.ndarray] = (
        SEMANTIC_CLASS_MAPPINGS[label_def].rgb
        if isinstance(label_def, str)
        else label_def
    )

    raw_buffer: np.ndarray = state.labels_buffer
    buffer: np.ndarray = np.empty((*raw_buffer.shape, 3), dtype=np.uint8)
    buffer[:, :, :] = label_def_[""].reshape((1, 1, 3))
    buffer[raw_buffer == 0, :] = label_def_["Floor/Ceil"]
    buffer[raw_buffer == 1, :] = label_def_["Wall"]

    cutoff = hud_padding[0]
    if cutoff:
        if cutoff < 0:
            row_is_all_zero = (raw_buffer == 0).all(axis=1)
            valid_row_indices = np.where(~row_is_all_zero)[0]

            if len(valid_row_indices) > 0:
                cutoff = hud_padding[0] = valid_row_indices[-1] + 1
        buffer[cutoff:] = 0

    if state.labels and "Self" in label_def_:
        for obj in state.labels[:-1]:
            buffer[raw_buffer == obj.value, :] = label_def_[obj.object_category]

        last_obj = state.labels[-1]
        if last_obj.object_category == "Player":
            buffer[raw_buffer == last_obj.value, :] = label_def_["Self"]
        else:
            buffer[raw_buffer == last_obj.value, :] = label_def_[
                last_obj.object_category
            ]
    else:
        for obj in state.labels:
            buffer[raw_buffer == obj.value, :] = label_def_[obj.object_category]

    return buffer


# Utility functions
def label2rgb(
    label_defs: defaultdict[str, int], color_map: Union[Colormap, str] = "turbo"
) -> defaultdict[str, np.ndarray]:
    """
    Convert label definitions to RGB color mappings using a matplotlib colormap.

    Arguments:
        label_defs (defaultdict[str, int]): A defaultdict mapping category names to label IDs.
        color_map (Union[Colormap, str]): A matplotlib colormap or colormap name to use for color generation.
                                          Default: "turbo".

    Returns:
        defaultdict[str, np.ndarray]: A defaultdict mapping category names to RGB color arrays.
                                     Unknown objects get the color corresponding to label ID 0.
                                     Each RGB color is a numpy array of shape (3,) with uint8 dtype.

    Example:

        >>> reserved = {"Floor/Ceil": 10, "Wall": 20, "Self": 250}
        >>> label_def = construct_label_def(Weapon=50, **reserved)
        >>> rgb_def = label2rgb(label_def, "viridis")
        >>> rgb_def["Wall"]  # Returns RGB array for label "Wall"
    """
    color_map_: Colormap = (
        getattr(cm, color_map) if isinstance(color_map, str) else color_map
    )

    assert label_defs.default_factory is not None
    default_rgb = np.asarray(
        color_map_(label_defs.default_factory())[:3], dtype=np.uint8
    )

    def return_default():
        return default_rgb

    return defaultdict(
        return_default,
        {
            obj_name: np.asarray(
                [c * 255 for c in color_map_(obj_val)[:3]], dtype=np.uint8
            )
            for obj_name, obj_val in label_defs.items()
        },
    )


def generate_default_mapping():
    def return_default():
        return 0

    categories = [
        "Artifact",
        "Powerup",
        "Explosive",
        "SFX",
        "Vegetation",
        "Hazard",
        "Decoration",
        "Light_source",
        "Breakable",
        "Interactive_object",
        "Player",
        "Monster",
        "Gibs",
        "Gore",
        "Armor",
        "Bridge",
        "Key",
        "Health",
        "Ammo",
        "Weapon",
    ]

    indices = [
        int(i)
        for i in np.linspace(0, 255, len(categories) + 4, endpoint=True, dtype=np.int32)
    ]

    mapping = defaultdict(
        return_default,
        {"Floor/Ceil": indices[1], "Wall": indices[2], "Self": indices[-1]},
    )

    for i, category in enumerate(categories, start=3):
        mapping[category] = indices[i]

    return register_semantic_mapping("default", mapping, label_cmap="jet")


# Global registry of semantic class mappings
SEMANTIC_CLASS_MAPPINGS: dict[str, SemanticClassDef] = {}
DEFAULT_MAPPING = generate_default_mapping()
