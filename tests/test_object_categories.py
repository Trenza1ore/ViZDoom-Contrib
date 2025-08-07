#!/usr/bin/env python3

# Tests for category mapping functionality (semantic segmentation).
# This test can be run as Python script or via PyTest

import glob
import os
import random

import vizdoom


def custom_categories_test(scenario: str = "scenarios/defend_the_line.wad"):
    print(f"Testing custom category mapping on {scenario}...")

    # Create game1 instance
    game1 = vizdoom.DoomGame()

    # Set up basic configuration
    game1.set_doom_scenario_path(scenario)
    game1.set_window_visible(False)
    game1.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
    game1.set_labels_buffer_enabled(True)
    game1.init()

    # Start a new episode
    random.seed("ViZDoom!")
    game1.set_seed(random.randrange(0, 256))
    game1.new_episode()

    default_mapping = vizdoom.get_default_category_mapping()
    seen_objects = set()
    seen_categories = set()

    # Move randomly for a while to generate some labels
    action_size = [0] * game1.get_available_buttons_size()
    for _ in range(2000):
        game1.make_action([random.random() > 0.5 for _ in action_size], 4)

        # Get the state and check labels
        state1 = game1.get_state()
        if state1 and state1.labels:
            labels = sorted(state1.labels, key=lambda label: label.object_name)
            for l1 in labels:
                if l1.object_category == "Self":
                    assert (
                        l1.object_name == "DoomPlayer"
                    ), f'Assigned "Self" to non-DoomPlayer object: {l1.object_name}'
                elif l1.object_category == "Unknown":
                    category_matches = [
                        category
                        for category in default_mapping
                        if l1.object_name.lower() in default_mapping[category]
                    ]
                    assert (
                        not category_matches
                    ), f'Assigned "Unknown" to known object: {l1.object_name} of category "{category_matches[0]}"'
                else:
                    category_matches = [
                        category
                        for category in default_mapping
                        if l1.object_name.lower() in default_mapping[category]
                    ] + ["Unknown"]
                    assert (
                        l1.object_name.lower() in default_mapping[l1.object_category]
                    ), f'Assigned "{l1.object_category}" to object: {l1.object_name} of category "{category_matches[0]}"'
                seen_objects.add(l1.object_name)
                seen_categories.add(l1.object_category)
        else:
            game1.new_episode()

    # Close the game
    game1.close()
    print(
        f"Seen objects: {', '.join(sorted(seen_objects))}\nSeen categories: {', '.join(sorted(seen_categories))}\n"
    )


def test_object_categories():
    scenario_pattern = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scenarios", "*.wad"
    )
    print(f"Finding scenarios in {scenario_pattern}")
    for scenario in sorted(glob.glob(scenario_pattern)):
        if any(multi_keyword in scenario for multi_keyword in ["multi", "cig"]):
            continue
        custom_categories_test(scenario)
    print("\nTest completed!")


if __name__ == "__main__":
    test_object_categories()
