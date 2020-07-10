import json
import codecs
import string

def encode(slots):
    requestables = ['address','name','postcode','phone','signature']
    encoded_array = [0 for i in range(5)]
    for slot in slots:
        if slot in requestables:
            encoded_array[requestables.index(slot)]=1
    return encoded_array


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


file_path = "train.json"
woz_json = json.load(codecs.open(file_path,"r","utf-8"))


dialogues = []
conversation = []

options = {}

##for idx in range(0,len(woz_json)):
##    num = 0
##    for conv in woz_json[idx]["dialogue"]:
##        if len(conv["system_transcript"])==0:
##            conversation.append(str(num)+" "+process_turn_hyp(conv["transcript"]))
##            num = num+1
##        else:
##            conversation.append(str(num)+" "+process_turn_hyp(conv["system_transcript"]))
##            num = num+1
##            conversation.append(str(num)+" "+process_turn_hyp(conv["transcript"]))
##            num = num+1

count = 0 
for idx in range(0,len(woz_json)):
    for conv in woz_json[idx]['dialogue']:
        dialogues.append({})
        dialogues[count]["dialog"] = process_turn_hyp(conv["transcript"]) + " " + process_turn_hyp(conv['system_transcript'])

        slots = [i[1] for i in conv['turn_label'] if i[0]=="request"]
        dialogues[count]["slots"] = slots
        count=count+1

##with open('conversation.txt','w') as f:
##    for item in conversation:
##        f.write("%s\n" % item)

with open("request-set-1.json","w") as write_file:
    json.dump(dialogues,write_file)
