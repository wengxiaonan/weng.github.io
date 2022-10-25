import json
from pathlib import Path
import random

import logging

from flask import Blueprint, jsonify, send_file

from psiturk.psiturk_config import PsiturkConfig
from psiturk.user_utils import PsiTurkAuthorization

logging.basicConfig(level=logging.DEBUG)
L = logging.getLogger(__name__)


config = PsiturkConfig()
config.load_config()
myauth = PsiTurkAuthorization(config) # if you want to add a password protected route use this

# explore the Blueprint
custom_code = Blueprint("custom_code", __name__, template_folder="templates", static_folder="static")


item_sequences_f = Path("/materials/all_items.json")
with item_sequences_f.open("r") as items_f:
    ITEM_SEQUENCES = json.load(items_f)


###############
# custom routes

@custom_code.route("/item_seq", methods=["GET"])
def get_item_seq():
    # TODO maybe not random sample, but ensure balanced sample
    item_seq = random.choice(ITEM_SEQUENCES)
    return jsonify(item_seq)
