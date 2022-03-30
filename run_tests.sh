#!/bin/bash
python -m unittest tests.test_replay_buffer
python -m unittest tests.test_gym_adapter
python -m unittest tests.test_configuration
python -m unittest tests.test_rssm
