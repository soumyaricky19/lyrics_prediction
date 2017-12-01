import sys
import re
import csv
import nltk
import math
import os
import operator
import time
import json
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from autocorrect import spell
from nltk.wsd import lesk as find_sense

uni="uni_"
bi="bi_"
tri="tri_"
unigram_bigram_trigram=bi

lemma="lemma_"
stem="stem_"
lemma_stem=lemma

pos="pos_"
lesk="lesk_"
pos_lesk=lesk

to_autocorrect=False

# lyrics_genre="lyrics.csv"
# lyrics_artist="songdata.csv"
# input_file_name=lyrics_artist

stop_words = nltk.corpus.stopwords.words('english')

prog_start_time=time.time()
def main(args):
    global bag_dir,input_file_name
    file = open("temp.txt", "r") 
    input_lyrics=file.read()
    file.close()
    if to_autocorrect:
        bag_dir="bags_"+unigram_bigram_trigram+lemma_stem+pos_lesk+"Autocorrect"
    else:
        bag_dir="bags_"+unigram_bigram_trigram+lemma_stem+pos_lesk+"NoAutocorrect"
    # input_lyrics="love"
    # input_lyrics="Ego so big, you must admit I got every reason to feel like I'm that bitch Ego so strong, if you ain't know I don't need no beat, I can sing it with piano"
    # input_lyrics="nigga"
    # input_lyrics="Alcohol, alcohol, alcohol, alcohol!"
    # print("Program started:"+elapsed_time())
    train()
    test(input_lyrics)

def preProcess(raw_text):  
    # convert to lower case
    processed = raw_text.lower()

    # ignore anything with [], example- [intro]
    processed = re.sub('\[.*\]', '',processed)

    # ignore irrelevant lines having words like verse and chorus
    processed=re.sub('.*(verse|chorus).*[\n]*', '', processed)

    # remove special characters
    processed=re.sub('[^\w\d\s\']+', ' ', processed)

    # remove multiple spaces
    processed=re.sub('[\r\t\f\v ]+', ' ',processed)

    # remove leading or trailing white space
    processed=re.sub('^\s+|\s+?$', '', processed)
    
    #autocorrect
    if to_autocorrect: 
        processed=autocorrect(processed)

    if pos_lesk == pos:
        processed = posTag(processed)
    else:
        processed = simplified_lesk(processed)

    if unigram_bigram_trigram == uni:
        processed=lemmatizeStem(processed)
        processed = removeStopWords(processed)

    return processed



def normalizePercentage(prob_map):
    per_map=[]
    total=0.0
    for key,value in prob_map:
        total+=value
    for key,value in prob_map:
        norm_value=(total-value)/total*100
        new_value=key,norm_value
        per_map.append(new_value)
    return per_map
    
def readBagFile(file_name):
    file=open(file_name,encoding="latin-1")
    lines=csv.reader(file)
    data=dict()
    # print(file_name)
    for row in lines:
        # row=row.strip()
        # print (row)
        if row:
            data[row[0]]=row[1]

    return data


def train():
    if not os.path.exists(bag_dir):
        os.makedirs(bag_dir)
    for file_name in os.scandir(bag_dir):
        if file_name.is_file():
            return
    print("Reading input file...")
    input_data=readInputFile("train.csv")
    print("Input file read")
    # print((input_data[0]))
    preProcessedData=[]
    print("Training started:",bag_dir)
    start_time=time.time()
    for i in range(len(input_data)):
        if i % 100 == 0 and i>0:
            time_diff=(time.time()-start_time)
            total_time=len(input_data)/i*time_diff
            print("Preprocessing "+str(i)+"/"+str(len(input_data))+", Elapsed time: "+str(round(time_diff,3))+", Remaining time: "+str(round(total_time-time_diff,3)))
            
        # p = Process(target=f, args=('bob',))
        # p.start()
        # p.join()
        data=preProcess(input_data[i][0]),input_data[i][1]
        preProcessedData.append(data)
    print(bag_dir+"_Training complete")
    print("Generating bag of words...")
    bags,genre_list=bagOfWords(preProcessedData)
    print("Bag of words generated")

    print("Writing bags...")
    writeBags(bags,genre_list)
    print("Bags written")

def test(input_text):
    # print("Starting test:"+elapsed_time())
    print("Testing started:"+bag_dir)
    correct=0.0
    wrong=0.0
    genre_list=[]
    for file_name in os.scandir(bag_dir):
        if file_name.is_file():
            genre_list.append(file_name.name.rstrip('csv').rstrip("."))
    
    c_matrix=dict()
    for x in genre_list:
        for y in genre_list:
            key=x,y
            c_matrix[key]=0
    
    if not input_text:
        print("Reading input file for test...")
        input_test_data=readInputFile("test.csv")
        print("Input file for test read")
    else:
        input_test_data=[[input_text]]
    n=0
    for test_row in input_test_data:
        n+=1
        if not input_text:
            print("Classifying data: "+str(n)+"/"+str(len(input_test_data)))

        words=preProcess(test_row[0])
        test_data=[[words,"test"]]
        test_bag_of_words,test_genre_list=bagOfWords(test_data)
        bag_of_words=[]
        total=0
        # print("Finding classes:"+elapsed_time())
        for genre in genre_list:
            file_name=bag_dir+"/"+genre+".csv"
            input_data=readBagFile(file_name)
            total+=len(input_data)
            bag_of_words.append(input_data)

        # print("Classes found:"+elapsed_time())
        prior=[]
        for i in range(len(genre_list)):
            prior.append(len(bag_of_words[i])/total)

        prob=(0.0 for x in range(len(genre_list)))

        prob_map=dict()
        i=0
        for bag in bag_of_words:
            likelihood=0.0
            total=0
            for key in bag:
                total+=int(bag[key])

            if unigram_bigram_trigram == uni:
                for word in test_bag_of_words[0]:
                    num_word=float(bag.get(str(word),0.0000000001))*test_bag_of_words[0][word]
                    smoothed_prob=num_word/total
                    likelihood+=math.log(smoothed_prob)
            elif unigram_bigram_trigram == bi:
                for j in range(len(test_bag_of_words[0])):
                    if j >= len(words)-1:
                        word=(words[j],"<end>")
                    else:
                        word=(words[j],words[j+1])
                    num_word=float(bag.get(str(word),0.0000000001))*test_bag_of_words[0][word]
                    smoothed_prob=num_word/total
                    likelihood+=math.log(smoothed_prob)
            elif unigram_bigram_trigram == tri:
                for j in range(len(test_bag_of_words[0])):
                    if j >= len(words)-2:
                        word=(words[j],"<end>","<end>")
                    else:
                        word=(words[j],words[j+1],words[j+2])
                    num_word=float(bag.get(str(word),0.0000000001))*test_bag_of_words[0][word]
                    smoothed_prob=num_word/total
                    likelihood+=math.log(smoothed_prob)
            # print(likelihood,genre_list[i])
            # print (i,prior)
            likelihood+=math.log(prior[i])
            prob_map[genre_list[i]]=likelihood
            i+=1

        sorted_prob = sorted(prob_map.items(), key=operator.itemgetter(1), reverse=True)
        norm_prob=normalizePercentage(sorted_prob)
        
        if not input_text:
            # print(sorted_prob)
            # print(test_row[1],norm_prob[0][0])
            if test_row[1] == norm_prob[0][0]:
                correct+=1
            else:
                wrong+=1
            key=test_row[1],norm_prob[0][0]
            c_matrix[key]=c_matrix[key]+1

    if not input_text:
        print(bag_dir+"_Accuracy: "+str(correct/(correct+wrong)*100))
        print(c_matrix)
    else:    
        # print("Printing result:"+elapsed_time())
        print(json.dumps(norm_prob))

def writeBags(bag_of_words,genre_list):
    i=0
    for bag in bag_of_words:
        # sorted_bag = sorted(bag.items(), key=operator.itemgetter(1), reverse=True)
        file_name=bag_dir+"\\"+genre_list[i]+".csv"
        genre_file  = open(file_name, "w")
        # writer = csv.writer(genre_file, delimiter='', quotechar='"', quoting=csv.QUOTE_ALL)
        writer = csv.writer(genre_file)
        for key,value in sorted(bag.items(), key=operator.itemgetter(1), reverse=True):
            row=key,value
            writer.writerow(row)
        i+=1
        genre_file.close()


def bagOfWords(data):
    genre_list=[]
    for x in data:
        genre=x[1]
        if genre not in genre_list:
            genre_list.append(genre)
    bag_of_words = [dict() for x in range(len(genre_list))]
    for x in data:
        word_list=x[0]
        genre=x[1]
        genre_idx=genre_list.index(genre)
        if unigram_bigram_trigram == uni:
            for word in word_list:
                if word in bag_of_words[genre_idx].keys():
                    bag_of_words[genre_idx][word]+=1
                else:
                    bag_of_words[genre_idx][word]=1
        elif unigram_bigram_trigram == bi:
            for i in range(len(word_list)):
                if i >= len(word_list)-1:
                    word=(word_list[i],"<end>")
                else:
                    word=(word_list[i],word_list[i+1])
                if word in bag_of_words[genre_idx].keys():
                    bag_of_words[genre_idx][word]+=1
                else:
                    bag_of_words[genre_idx][word]=1
        elif unigram_bigram_trigram == tri:
            for i in range(len(word_list)):
                if i >= len(word_list)-2:
                    word=(word_list[i],"<end>","<end>")
                else:
                    word=(word_list[i],word_list[i+1],word_list[i+2])
                if word in bag_of_words[genre_idx].keys():
                    bag_of_words[genre_idx][word]+=1
                else:
                    bag_of_words[genre_idx][word]=1
    
    # print (len(bag_of_words))
    # print (len(genre_list))

    return bag_of_words,genre_list

def autocorrect(text):
    processed_words=""
    sentences=text.split("\n")
    for sentence in sentences:
        # words=sentence.split(" ")
        words=re.findall(r"\b[\w\d\']+\b", sentence)
        for word in words:
            corrected_word=spell(word)
            processed_words+=corrected_word+" "
        processed_words=processed_words.rstrip()
        processed_words+="\n"
    return processed_words

def lemmatizeStem(text):
    processed_text=[]
    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    for word_sense,tag in text:
        word=""
        sense=""
        key=""
        if pos_lesk == lesk:
            word_list=word_sense.split(".")
            word=word_list[0]
            sense=word_list[2]
        else:
            word=word_sense

        lemmaORstem=''
        if lemma_stem == lemma:
            lemmaORstem=wordnet_lemmatizer.lemmatize(word, pos=tag)
        elif lemma_stem == stem:
            lemmaORstem=porter_stemmer.stem(word)

        if pos_lesk == lesk:
            key=lemmaORstem+"."+sense,tag
        else:
            key=lemmaORstem,tag
        processed_text.append(key)
    return processed_text

def removeStopWords(input_text):
    output_text=[]
    for word_sense,tag in input_text:
        word=""
        sense=""
        if pos_lesk == lesk:
            word_list=word_sense.split(".")
            word=word_list[0]
            sense=word_list[1]
        else:
            word=word_sense
        if word not in stop_words:
            if pos_lesk == lesk:
                output_text.append((word+"."+sense,tag))
            else:
                output_text.append((word,tag))
    return output_text

def posTag(text):
    processed_text=[]
    sentences=text.split("\n")
    for sentence in sentences:
        word_list = re.findall(r"\b[\w\d\']+\b", sentence)
        with_tags=nltk.pos_tag(word_list)
        for word_tag in with_tags:
            word=word_tag[0]
            pos_tag=word_tag[1]
            wn_tag=get_wordnet_pos(pos_tag)
            if not wn_tag:
                wn_tag='n'
            key=word,wn_tag
            processed_text.append(key)
    # print (processed)
    return processed_text

def simplified_lesk(text):
    processed_text=[]
    sentences=text.split("\n")
    for sentence in sentences:
        best_sense=0
        max_overlap=0
        context=re.findall(r"\b[\w\d\']+\b", sentence)
        with_pos_tags=nltk.pos_tag(context)
        for word_pos_tag in with_pos_tags:
            word=word_pos_tag[0]
            pos_tag=word_pos_tag[1]
            wn_tag=get_wordnet_pos(pos_tag)
            if wn_tag:
                best_sense=find_sense(context, word, wn_tag)

            if best_sense and wn_tag:
                name=best_sense._name
                tag=best_sense._pos
                key=name,tag
            else:
                name=word+".n.01"
                tag="n"
                key=name,tag

            processed_text.append(key)
    return processed_text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

def readInputFile(file_name):
    file=open(file_name,encoding="latin-1")
    lines=csv.reader(file)
    is_header=True
    data=[]
    for row in lines:
        # print(row)
        if is_header:
            is_header=False
        else:
            # print (row)
            if len(row)==2 and row[1] and row[0]:
                data.append([row[0],row[1]])
        
    return data

def elapsed_time():
    return str(time.time()-prog_start_time)

main(sys.argv)