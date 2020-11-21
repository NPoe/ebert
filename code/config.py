import os

SRC_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
RESOURCE_DIR = os.path.join(ROOT_DIR, "resources")
MAPPERS_DIR = os.path.join(ROOT_DIR, "mappers")
ROBERTA_START_CHAR = "Ġ"
XLNET_START_CHAR = "▁"

