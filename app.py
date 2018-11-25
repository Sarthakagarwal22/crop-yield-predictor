import flask 
from google.cloud import texttospeech
from flask import request
import datetime
from sklearn.externals import joblib
import numpy as np
app = flask.Flask(__name__)

class ModelLoad:
	models = []
	def load(self):
		for i in range(0,105):
			ModelLoad.models.append(joblib.load('mods/crop'+str(i)+'.pkl'))


@app.route('/')
def hello_world():
	ml = ModelLoad()
	ml.load()
	return flask.render_template('index.html', has_result=False)

@app.route('/output')
def output():
	lat = request.args.get('lat')
	lng = request.args.get('lng')
	now = datetime.datetime.now()
	models = []
	avg = [1.284837797231591,2.18481794561088,0.7801713213236904,6.216615908824843,1.1372425381994007,26.092243648704116,1.7606134816203758,1.3400001866104088,5.102352996380728,6.049771085483849,5.84149184149184,1.1868708689922471,0.716424605157452,7.678417037507947,10.467539706167004,25.069836612976804,0.11822640091437589,7.021800133131402,9.703013375566464,0.11101759215570146,0.5026502779234476,0.6960416485278671,8.47936278549632,6.764081874402016,4277.609020816736,0.7088075516647366,9.510369595198457,0.8329151746996777,0.4866098092461242,2.141185541841432,0.49511772196744025,0.9734623687981889,1.457217816056103,5.540574162758938,2.981379202343344,8.20212589784688,0.8147619749481216,22.55141746297082,1.2277057447128608,1.1205653860912863,0.4405722075980747,6.582940779779704,1.0058201058201057,1.068147635460453,8.049757977278228,8.114555455371264,0.7663445733926881,0.7489385640273843,0.6782316292545555,7.932812276256422,0.8153196709808701,0.5599996711476566,2.897758372758643,5.3270940057579494,0.675881456986058,5.034269548032068,0.4331095830546137,0.34959373453209264,0.3812452567998751,1.0288528223806994,11.236168212726147,9.677284937184183,0.6295445783860036,0.8178596036400303,3.6388348883111528,0.4877975127805305,5.498244031403952,1.9477810615658437,33.598246565438636,1.3633308948353207,0.5958030882603939,13.996944130905732,8.271700529347385,12.469846102047127,10.349474095393967,0.6402890195906491,1.1934159323881648,1.2438567540822327,0.7514613768386398,8.457873546912566,1.954143007749865,1.1074301252257843,1.1859109662121985,0.5316615949336372,0.5540506556671749,1.174486172430886,10.267325516590827,0.496053976127272,0.6341818678036721,1.0948258193129161,229.4709095607241,0.8330561691380823,8.075504708665594,18.7852712203163,2.3953309424432776,1.2380849082259462,10.473163172527192,1.235359812679969,2.84369822576111,0.6034304737565055,0.4695167765090307,0.9310659245403735,1.9678562533657344,0.25055161341874366,2.1214633763348565]
	classes = ["aarkanat (sansaadhit)",
				"supaaree",
				"arahar / tur",
				"akharot (kachche)",
				"baajare",
				"kela",
				"jau",
				"sem",
				"beens aur matar (sabjee)",
				"bhindee",
				"karela",
				"kaalee mirch",
				"kaala chana",
				"laukee",
				"baingan",
				"gobhee",
				"ilaayachee",
				"gaajar",
				"kaajoo",
				"chashaiwnut sansaadhit",
				"chashaiwnut kachche",
				"kaastar beej",
				"gobhee",
				"khatte phal",
				"naariyal",
				"kofee",
				"cholochosi",
				"kand-spaaks any",
				"dhaniya",
				"kapaas (phaaha)",
				"lobiya (lobi)",
				"dhol ka chhadee",
				"sookhee mirch",
				"sookha adarak",
				"lahasun",
				"adarak",
				"graam",
				"angoor",
				"moongaphalee",
				"gvaar beej",
				"chane kee daal",
				"jaik phal",
				"jobstair",
				"jvaar",
				"joot",
				"joot aur mesta",
				"kaapas",
				"khaisari",
				"korr",
				"neemboo",
				"masoor",
				"alasee ka beej",
				"makka",
				"aam",
				"masoor",
				"maist",
				"moon (green graam)",
				"keet",
				"naijar beej",
				"kul tilahan",
				"pyaaj",
				"naarangee",
				"any rabee daalen",
				"any anaaj aur milet",
				"any taaja phal",
				"any khareeph daalen",
				"any sabjiyaan",
				"dhaan",
				"papeeta",
				"matar aur sem (daalen)",
				"pairill",
				"anaanaas",
				"pom phal",
				"pom grainet",
				"aaloo",
				"kul daalen",
				"raagee",
				"raajamash kholar",
				"raipised aur sarason",
				"raidish",
				"chaaval",
				"chaavalabeen (naagaadal)",
				"rabar",
				"kusum",
				"samai",
				"sannhamp",
				"cheekoo",
				"til",
				"chhote baajara",
				"soyaabeen",
				"ganna",
				"soorajamukhee",
				"shakarakand",
				"taipioka",
				"chaay",
				"tambaakoo",
				"tamaatar",
				"kul khaadyaann",
				"haldee",
				"shalajam",
				"udad",
				"varagu",
				"gehoon",
				"any vividh daalon",
				"any tilahan"
				]
	models = ModelLoad.models
	ans = []
	for i in range(0,105):
		ans.append(models[i].predict([[now.month, float(lat), float(lng)]])/avg[i])
	text="Humari gannit ke anusaar, iss maheene aap "+str(classes[np.argmax(ans)])+" ko ugaein. Apki fasal ka paidavaar "+str("{:.2f}".format(np.max(ans)))+" quintal prati hectare hone ki sambhaavnaa hai"
	return flask.render_template('output.html', has_result=False,text=text)

# @app.route('/t2s', methods=['POST','GET'])
# def t2s():
# 	client = texttospeech.TextToSpeechClient()
	

# 	synthesis_input = texttospeech.types.SynthesisInput(text="Latitude is " + lat + "and Longitude is " + lng)

# 	voice = texttospeech.types.VoiceSelectionParams(
#     	language_code='en-US',
#     	ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

# 	audio_config = texttospeech.types.AudioConfig(
# 	    audio_encoding=texttospeech.enums.AudioEncoding.MP3)

# 	response = client.synthesize_speech(synthesis_input, voice, audio_config)

# 	with open('static/output.mp3', 'wb') as out:
# 	    out.write(response.audio_content)
# 	    print('Audio content written to file "output.mp3"')

# 	return 'Audio content written to file "output.mp3"'   
