import json
import codecs
import string


def process_turn_hyp(transcription):
    """
    Returns the clean (i.e. handling interpunction signs) string for the given language. 
    """
    exclude = set(string.punctuation)
    exclude.remove("'")
    exclude.remove("?")

    transcription = ''.join(ch for ch in transcription if ch not in exclude)

    transcription = transcription.lower()
    transcription = transcription.replace(u"’", "'")
    transcription = transcription.replace(u"‘", "'")
    transcription = transcription.replace("don't", "dont")
    transcription = transcription.replace("'", "")
    
    return transcription


file_path = "data/test.json"
woz_json = json.load(codecs.open(file_path,"r","utf-8"))

dialogues = []
count = 0

for idx in range(0,len(woz_json)):
    prev_ques = ""
    for conv in woz_json[idx]['dialogue']:
        dialogues.append({})

##      get the final question asked by the system
        if ('?' in conv["system_transcript"]):
            prev_ques = conv["system_transcript"]

        if ('no' in conv["transcript"]):
            dialogues[count]["dialog"] = process_turn_hyp(prev_ques)
        else:
            dialogues[count]["dialog"] = process_turn_hyp(conv["transcript"]) + " " + process_turn_hyp(conv["system_transcript"])

        dialogues[count]['food'] = []
        dialogues[count]['area'] = []
        dialogues[count]['name'] = []
        dialogues[count]['price range'] = []

        for i in conv['belief_state']:
            if (i['act']=='inform'):
                dialogues[count][i['slots'][0][0]].append(i['slots'][0][1])
        count = count+1

with open("test/inform-option1-test.json","w") as write_file:
    json.dump(dialogues,write_file)
