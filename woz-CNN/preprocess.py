import json
import codecs
from copy import deepcopy
import string

def process_turn_hyp(transcription):
    """
    Returns the clean (i.e. handling interpunction signs) string for the given language. 
    """
    exclude = set(string.punctuation)
    exclude.remove("'")

    transcription = ''.join(ch for ch in transcription if ch not in exclude)

    transcription = transcription.lower()
    transcription = transcription.replace(u"’", "'")
    transcription = transcription.replace(u"‘", "'")
    transcription = transcription.replace("don't", "dont")
    transcription = transcription.replace("'", "")
    
    return transcription

def process_woz_dataset(woz_dialogue):
    null_bs = {}
    null_bs["food"] = "none"
    null_bs["price range"] = "none"
    null_bs["area"] = "none"
    null_bs["request"] = []
    informable_slots = ["food", "price range", "area"]
    pure_requestables = ["address", "phone", "postcode"]

    prev_belief_state = deepcopy(null_bs)
    dialogue_representation = []

    current_req = [""]
    current_conf_slot = [""]
    current_conf_value = [""]

    lp = {}

    for idx, turn in enumerate(woz_dialogue):
        current_DA = turn["system_acts"]
        current_req = []
        current_conf_slot = []
        current_conf_value = []

        for each_da in current_DA:

            if each_da in informable_slots:
                current_req.append(each_da)
            elif each_da in pure_requestables:
                current_conf_slot.append("request")
                current_conf_value.append(each_da)
            else:
                if type(each_da) is list:
                    current_conf_slot.append(each_da[0])
                    current_conf_value.append(each_da[1])

        if not current_req:
            current_req = [""]
        if not current_conf_slot:
            current_conf_slot = [""]
            current_conf_value = [""]

        current_transcription = turn["transcript"]
        current_transcription = process_turn_hyp(current_transcription)        

        read_asr = turn["asr"]
        current_asr = []
        for (hyp,score) in read_asr:
            current_hyp = process_turn_hyp(hyp)
            current_asr.append((current_hyp,score))

        old_trans = current_transcription

        exclude = set(string.punctuation)
        exclude.remove("'")

        current_transcription = ''.join(ch for ch in current_transcription if ch not in exclude)
        current_transcription = current_transcription.lower()
        
        current_labels = turn["turn_label"]
        current_bs = deepcopy(prev_belief_state)
    
        if "request" in prev_belief_state:
            del prev_belief_state["request"]

        current_bs["request"] = []

        for label in current_labels:
            (c_slot,c_value) = label

            if c_slot in informable_slots:
                current_bs[c_slot] = c_value

            elif c_slot == "request":
                current_bs["request"].append(c_value)

        curr_lab_dict = {}
        for x in current_labels:
            if x[0]!="request":
                curr_lab_dict[x[0]] = x[1]

        dialogue_representation.append(((current_transcription, current_asr), current_req, current_conf_slot, current_conf_value, deepcopy(current_bs), deepcopy(prev_belief_state)))        

        print("====", current_transcription, "current bs", current_bs, "past bs", prev_belief_state, "this turn update", curr_lab_dict)
        prev_belief_state = deepcopy(current_bs)

    return dialogue_representation


file_path = "train.json"
woz_json = json.load(codecs.open(file_path,"r","utf-8"))

dialogues = []
training_turns = []

dialogue_count = len(woz_json)

for idx in range(0,dialogue_count):

    current_dialogue = process_woz_dataset(woz_json[idx]["dialogue"])
    
