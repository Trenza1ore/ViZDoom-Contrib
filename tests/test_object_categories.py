#!/usr/bin/env python3

# Tests for category mapping functionality (semantic segmentation).
# This test can be run as Python script or via PyTest

import glob
import os

import vizdoom


def custom_categories_test(scenario: str = "scenarios/defend_the_line.wad"):
    print(f"Testing custom category mapping on {scenario}...")

    # Create game1 instance
    game1 = vizdoom.DoomGame()

    # Set up basic configuration
    game1.set_window_visible(False)
    game1.set_doom_scenario_path(scenario)
    game1.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
    game1.set_labels_buffer_enabled(True)

    # Create second game instance before game1.init()
    game2 = vizdoom.DoomGame()
    game2.set_window_visible(False)
    game2.set_doom_scenario_path(scenario)
    game2.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
    game2.set_labels_buffer_enabled(True)

    # Define custom category mapping
    custom_mapping = {
        "CustomMonster": {"doomimp", "demon", "cacodemon", "baronofhell"},
    }

    # Set the custom mapping
    game1.set_category_mapping(custom_mapping)

    # Initialize the games
    game1.init()
    game2.init()

    # Start a new episode
    game1.set_seed(123)
    game2.set_seed(123)
    game1.new_episode()
    game2.new_episode()

    # Take a few actions to generate some labels
    for i in range(100):
        game1.make_action([0, 0, 0, 0, 0, 0, 0, 0, 0])  # No action
        game2.make_action([0, 0, 0, 0, 0, 0, 0, 0, 0])  # No action

        # Get the state and check labels
        state1 = game1.get_state()
        state2 = game2.get_state()
        if state1 and state2 and state1.labels and state2.labels:
            labels = sorted(state1.labels, key=lambda label: label.object_name)
            labels2 = sorted(state2.labels, key=lambda label: label.object_name)
            for l1, l2 in zip(labels, labels2):
                assert l1.object_name == l2.object_name
                if l2.object_category != "Self":
                    assert l1.object_category == game1.get_category_for_class(
                        l1.object_name
                    )
                    assert l2.object_category == game2.get_category_for_class(
                        l2.object_name
                    )
                if l2.object_category != "Unknown":
                    assert l1.object_category != l2.object_category

    # Create third game instance after game1.init()
    game3 = vizdoom.DoomGame()

    # Set up basic configuration
    game3.set_window_visible(False)
    game3.set_doom_scenario_path(scenario)
    game3.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
    game3.set_labels_buffer_enabled(True)
    game3.init()

    # Reset the custom mapping of game1
    game1.set_category_mapping({})

    # Start a new episode
    game1.set_seed(123)
    game3.set_seed(123)
    game1.new_episode()
    game3.new_episode()

    # Take a few actions to generate some labels
    for i in range(100):
        game1.make_action([0, 0, 0, 0, 0, 0, 0, 0, 0])  # No action
        game3.make_action([0, 0, 0, 0, 0, 0, 0, 0, 0])  # No action

        # Get the state and check labels
        state1 = game1.get_state()
        state3 = game3.get_state()
        if state1 and state3 and state1.labels and state3.labels:
            labels = sorted(state1.labels, key=lambda label: label.object_name)
            labels3 = sorted(state3.labels, key=lambda label: label.object_name)
            for l1, l3 in zip(labels, labels3):
                assert l1.object_name == l3.object_name
                if l3.object_category != "Self":
                    assert l1.object_category == game1.get_category_for_class(
                        l1.object_name
                    )
                    assert l3.object_category == game3.get_category_for_class(
                        l3.object_name
                    )
                assert l1.object_category == l3.object_category

    # Close the games
    game1.close()
    game2.close()
    game3.close()


def test_object_categories():
    scenario_pattern = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scenarios", "*.wad"
    )
    print(f"Finding scenarios in {scenario_pattern}")
    for scenario in glob.glob(scenario_pattern):
        if any(multi_keyword in scenario for multi_keyword in ["multi", "cig"]):
            continue
        custom_categories_test(scenario)
    print("\nTest completed!")


if __name__ == "__main__":
    test_object_categories()
