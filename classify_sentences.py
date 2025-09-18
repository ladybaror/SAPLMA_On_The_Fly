import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#import ast


def correct_str(str_arr):
    val_to_ret = str_arr.replace("[array(", "").replace("dtype=float32)]", "").replace("\n","").replace(" ","").replace("],","]").replace("[","").replace("]","")
    return val_to_ret

layer_to_test = -1 #-1 #-4 #-8 #-16
model_to_use = "SmolLM-1.7B" #"125m" #"6.7b" #"6.7b" "2.7b" "1.3b" "350m"
num_of_runs = 10

use_logistic_regression = False
use_random_labels = False
accuracy_on_train = False
use_median_length_as_label = False
# # Read the CSV file
# df1 = pd.read_csv('resources\\embeddings_with_labels_capitals6.7b_5fromend_rmv_period.csv')
# df2 = pd.read_csv('resources\\embeddings_with_labels_inventions6.7b_5fromend_rmv_period.csv')
# df3 = pd.read_csv('resources\\embeddings_with_labels_elements6.7b_5fromend_rmv_period.csv')
# df4 = pd.read_csv('resources\\embeddings_with_labels_animals6.7b_5fromend_rmv_period.csv')
# df5 = pd.read_csv('resources\\embeddings_with_labels_facts6.7b_5fromend_rmv_period.csv')
# df6 = pd.read_csv('resources\\embeddings_with_labels_companies6.7b_5fromend_rmv_period.csv')
#test_df = pd.read_csv('resources\\embeddings_with_labels_math6.7b_5fromend_rmv_period.csv')

dataset_names = ["capitals", "inventions", "elements", "animals", "facts", "companies"] # ["capitals", "inventions", "elements", "animals", "facts", "companies"]#["generated", "capitals", "inventions", "elements", "animals", "facts", "companies"]
#["capitals", "inventions", "elements", "animals", "facts", "companies"
datasets = []
for dataset_name in dataset_names:
    #datasets.append(pd.read_csv('resources\\embeddings_with_labels_'+dataset_name+'6.7b_5fromend_rmv_period.csv'))
    datasets.append(pd.read_csv("embeddings_3_sentences\\3_sentence_with_tokens_"+dataset_name+"_Smollm_1.7B_12.csv", encoding='latin1'))
results = []
for i in range(len(dataset_names)):
    test_df = datasets[i]
    dfs_to_concatenate = datasets[:i] + datasets[i + 1:]
    train_df = pd.concat(dfs_to_concatenate, ignore_index=True)

    tot_acc = 0
    for j in range(num_of_runs):
        # test_df = pd.read_csv('resources\\embeddings_with_labels_grammar6.7brmv_period.csv')
        # df5 = pd.read_csv('resources\\embeddings_with_labels_colors6.7brmv_period.csv')
        # df = pd.concat([df1,df2,df5,df3,df6,df4], ignore_index=True)
        # df = df[df['next_id'] == 4]  #only those in which the next token is supposed to be '.'
        #
        # test_df = pd.read_csv('resources\\gen_text_embed_label2.csv')
        # train_df = pd.read_csv('resources\\gen_text_embed_label3.csv')

        # Split the data into train and test sets
        #train_df, test_df = train_test_split(datasets[0], test_size=0.2, random_state=42)

        #np.fromstring(train_df['embeddings'].tolist()[0].replace("[array(", "").replace("dtype=float32)]", "").replace("\n","").replace(" ","").replace("],","]"), sep=',')
        # Extract the embeddings and labels from the train and test sets
        train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_df['embeddings'].tolist()])
        test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_df['embeddings'].tolist()])
        if use_median_length_as_label:
            # Compute the median sentence length for both training and testing datasets
            median_train_length = np.median(train_df['statement'].str.len())
            median_test_length = np.median(test_df['statement'].str.len())
            # Set labels based on the sentence length in relation to the median
            train_df['label'] = np.where(train_df['statement'].str.len() > median_train_length, 1, 0)
            test_df['label'] = np.where(test_df['statement'].str.len() > median_test_length, 1, 0)
        train_labels = np.array(train_df['label'])
        test_labels = np.array(test_df['label'])

        if use_random_labels:
            # Generate random labels of 0 and 1
            train_labels = np.random.randint(2, size=len(train_df))
            test_labels = np.random.randint(2, size=len(test_df))



        # train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_df['embed'].tolist()])
        # test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_df['embed'].tolist()])
        # train_labels = np.array(train_df['truth_label'])
        # test_labels = np.array(test_df['truth_label'])


        # Define the neural network model
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=train_embeddings.shape[1])) #change input_dim to match the number of elements in train_embeddings...
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        if use_logistic_regression:
            model = Sequential()
            model.add(Dense(1, activation='sigmoid', input_dim=train_embeddings.shape[1]))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile the model
        #from tensorflow.keras.models import load_model
        #model = load_model("resources\\classifier_all.h5")

        # Train the model
        model.fit(train_embeddings, train_labels, epochs=5, batch_size=32, validation_data=(test_embeddings, test_labels))
        loss, accuracy = model.evaluate(test_embeddings, test_labels)
        if accuracy_on_train:
            loss, accuracy = model.evaluate(train_embeddings, train_labels)
        tot_acc += accuracy
        
        model.save("h5\\classifier_3_sentences_"+dataset_names[i]+"_"+str(j)+"_.h5")
    results.append((dataset_names[i], tot_acc/num_of_runs))
print(results)
# Extract the second item from each tuple and put it in a list
acc_list = [t[1] for t in results]
# Calculate the average of the numbers in the list
avg_acc = sum(acc_list) / len(acc_list)
print(avg_acc)

#results
#all 5 using 2.7b with 100 ephocs 0.6727 (went up to 0.7072 after 7 epochs (and also after 12)
#only grammar using 2.7 with 100 ephocs 0.6339, but reached 0.7804 after 7 epochs
#only capitals using 2.7b 0.9041 after 30 epochs
#all but grammar 0.7 after 30
#new all with cities (no grammar): 0.7746/0.7684
#cities only 0.8775
#facts only 0.8000

#all but cities, for cities 0.6069
#all but facts, for facts 0.5969
#all but colors, for colors 0.5634
#all but inventions, for inventions 0.5256
#train on cities, test on facts: 0.5969
#train on facts, test on cities: 0.6263 (0.69 at 7 and 3, 0.6503 at 5)

#6.7b
#train on cities(1000), test on facts: 0.7569
#train cities(1000) and facts, test 20% 0.7547 (a little more)
#train facts test cities: 0.6630
#all (5) 0.7148/0.72+ (100) 0.74
#all but color, for colors 0.6791/0.6
#all but cities, for cities 0.6310/0.6570
#cities only (1000) 0.84
#facts only 0.8154/0.7385/0.7538/0.80 (100 epochs)

#6.7b rmv_period
#cities only(2000) 0.8950
#train on cities(2000), test on facts: 0.7138/0.6462
#train facts test cities: 0.7020/0.6775/0.5925/0.6105
#train facts+colors test cities: 0.6610/0.7045/0.6825
#train cities(2000)+facts+colors test 20%: 0.8459/0.8574/0.8131
#all (5) 0.7803/0.7594/0.7761 (100) 0.7914/0.7969
#all but cities, for cities 0.7465/0.6000/0.7580/0.6740/0.7325
#all but color, for colors 0.6045/0.7015/0.556/0.6343/0.6642/0.6604
#facts only (100 epochs) 0.7077/0.7231/0.7231/0.7692
#all (including grammar(1000)) 10  0.7486
#all but grammar, for grammar 10 0.5150

#6.7b rmv_period layer-5
#cities for facts 0.7754/0.7477/0.7754
#only cities(3000): (5) 0.8200/0.8760/0.8733/0.8717 (20) 0.8433/0.8800 (100) 0.8717
#facts for cities: 0.7240/0.7440/0.6797/0.7177
#inventions for cities: 0.6087/0.7213/0.7053/0.6953
#inventions+facts for cities: 0.7687/0.7600/0.7607/0.7307/0.7700
#all but cities, for cities: 0.7287/0.7630/0.7193/0.6153/0.7553/0.7563/0.6900
#all but color, for colors: 0.5784/0.5410/0.5373/0.5485/0.5224/0.5410
#all but facts for facts: 0.7385/0.6523/0.7323/0.7015
#all but invensions for inventions: 0.5560/0.5720/0.5510
#all (exc. grammar) (5): 0.7889/0.7889/0.7780 (100) 0.8096
#cities+inventions+facts (5): 0.8231/0.8208/0.7954/0.8301

#6.7b rmv_period layer-5 ***NEW
# all (cities(3000), inventions, elements, animals): (5) 0.7326/0.7271/0.7116 (100) 0.7605/0.7457
# all but cities for cities: 0.5663/0.5867/0.7207
# animals and inventions for cities: 0.5643/0.7363/0.5703/0.6400/0.6430
# cities(3000), inventions, animals: (5:) 0.7745/0.7725/0.7794/0.7894  (100:) 0.7914/0.8004
# all but inventions for inventions: 0.5540/0.5490/0.5410
# all but elements for elements: 0.5492/0.5548
# all but animals for animals: 0.6240/0.6468
# cities for animals: 0.6131/0.6429
# animals for cities: 05890/0.5367/0.5720
# added facts. All but facts for facts: 0.7292/0.7692/0.7785/0.7938/0.7754
# added also colors. All but facts for facts:0.7385/0.7325

#****************new!!!*********
#all: (5) 0.7646/0.7492 (100) 0.7636
#all but cities for cities: 0.7119/0.8402/0.6920/0.8368
#all but inventions for inventions: 0.5434/0.5285/0.5582
#all but elements for elements: 0.6065/0.6591/0.6237
#all but animals for animals: 0.5863/0.5516/0.5645
#all but facts for facts: 0.6819/0.6525/0.6313/0.6770/0.6069/0.6542
#only cities: 0.8733/0.9144/0.9178
#only inventions: 0.6648/0.6420/0.5966
#only facts: 0.6829/0.6748
#cities for facts: 0.6770/0.6900
#only elements: 0.6452/0.6774/0.6237
#only animals: 0.6683/0.6782/0.6584


#ADDED MATH+COMPANIES
#only math: 0.7037/0.6157/0.7685
#only companies: 0.8833/0.8500/0.8500
#capitals for companies: 0.6283/0.7392/0.6833
#companies for capitals: 0.5350/0.5775/0.5741
#companies+math for capitals: 0.6996/0.5898/0.5384
#all but capitals for capitals: 0.7723/0.7956
#all but companies for companies: 0.7300/0.7392
#all but math for math: 0.5056/0.5000
#**** removed math! ****
#all but animals for animals: 0.5833/0.6032
#all but facts for facts: 0.6835/0.6639
#all but elements for elements: 0.6032/0.5925/0.5968/0.5903
#all but inventions for inventions: 0.6199/0.6039/0.6313/0.6381
#all: 0.7781/0.7864/0.7749/0.7929/0.7847 (100) 0.7913
#with generated: 0.663776057107108 [('generated', 0.6130267977714539), ('capitals', 0.7427983283996582), ('inventions', 0.6484017968177795), ('elements', 0.5838709473609924), ('animals', 0.6170634627342224), ('facts', 0.6721044182777405), ('companies', 0.7691666483879089)]

########### summary layer: -5
#0.6479146182537079 [('capitals', 0.8209876418113708), ('inventions', 0.6073059439659119), ('elements', 0.5892472863197327), ('animals', 0.5714285969734192), ('facts', 0.6068515777587891), ('companies', 0.6916666626930237)]
#0.6652063429355621 [('capitals', 0.8155006766319275), ('inventions', 0.6061643958091736), ('elements', 0.5849462151527405), ('animals', 0.5773809552192688), ('facts', 0.6655791401863098), ('companies', 0.7416666746139526)]
#0.6712132493654887 [('capitals', 0.7709190845489502), ('inventions', 0.5878995656967163), ('elements', 0.603225827217102), ('animals', 0.6339285969734192), ('facts', 0.6704730987548828), ('companies', 0.7608333230018616)]

########back to layer:-1 ("generated", "capitals", "inventions", "elements", "animals", "facts", "companies")
#with generated:
#[('generated', 0.6398467421531677), ('capitals', 0.7647462487220764), ('inventions', 0.517123281955719), ('elements', 0.5881720185279846), ('animals', 0.5972222089767456), ('facts', 0.6541598439216614), ('companies', 0.7425000071525574)]
#0.6433957644871303
#[('generated', 0.5938697457313538), ('capitals', 0.7565158009529114), ('inventions', 0.6221461296081543), ('elements', 0.5838709473609924), ('animals', 0.6121031641960144), ('facts', 0.6982055306434631), ('companies', 0.7524999976158142)]
#0.6598873308726719

#without generated:
# 0.6507926285266876 [('capitals', 0.7990397810935974), ('inventions', 0.5570776462554932), ('elements', 0.5774193406105042), ('animals', 0.6220238208770752), ('facts', 0.6525285243988037), ('companies', 0.6966666579246521)]
# 0.653127908706665 [('capitals', 0.7846364974975586), ('inventions', 0.577625572681427), ('elements', 0.5827956795692444), ('animals', 0.5922619104385376), ('facts', 0.6639478206634521), ('companies', 0.7174999713897705)]
# 0.6309391458829244 [('capitals', 0.7551440596580505), ('inventions', 0.5742009282112122), ('elements', 0.5677419304847717), ('animals', 0.5922619104385376), ('facts', 0.6329526901245117), ('companies', 0.6633333563804626)]

############### -16
# [('generated', 0.5785440802574158), ('capitals', 0.7352537512779236), ('inventions', 0.6278538703918457), ('elements', 0.5827956795692444), ('animals', 0.5694444179534912), ('facts', 0.631321370601654), ('companies', 0.7933333516120911)]
# 0.6455066459519523

# 0.6563580632209778 [('capitals', 0.7688614726066589), ('inventions', 0.628995418548584), ('elements', 0.5720430016517639), ('animals', 0.5892857313156128), ('facts', 0.6247960925102234), ('companies', 0.7541666626930237)]
# 0.6379708548386892 [('capitals', 0.675583004951477), ('inventions', 0.6369863152503967), ('elements', 0.57419353723526), ('animals', 0.5734127163887024), ('facts', 0.6084828972816467), ('companies', 0.7591666579246521)]
# 0.6600262125333151 [('capitals', 0.7860082387924194), ('inventions', 0.6541095972061157), ('elements', 0.5473118424415588), ('animals', 0.5773809552192688), ('facts', 0.6378466486930847), ('companies', 0.7574999928474426)]

######################3 -8
# 0.6877709229787191 [('capitals', 0.7805212736129761), ('inventions', 0.6541095972061157), ('elements', 0.6225806474685669), ('animals', 0.579365074634552), ('facts', 0.6900489330291748), ('companies', 0.800000011920929)]
# 0.6883626282215118 [('capitals', 0.8004115223884583), ('inventions', 0.6929223537445068), ('elements', 0.6204301118850708), ('animals', 0.5853174328804016), ('facts', 0.6802610158920288), ('companies', 0.7508333325386047)]
# 0.6895275513331095 [('capitals', 0.8079560995101929), ('inventions', 0.6666666865348816), ('elements', 0.6204301118850708), ('animals', 0.5753968358039856), ('facts', 0.6900489330291748), ('companies', 0.7766666412353516)]

####################### -12
# 0.7171050806840261 [('capitals', 0.8525377511978149), ('inventions', 0.7351598143577576), ('elements', 0.6107527017593384), ('animals', 0.5833333134651184), ('facts', 0.6916802525520325), ('companies', 0.8291666507720947)]
# 0.7046360671520233 [('capitals', 0.777091920375824), ('inventions', 0.75), ('elements', 0.624731183052063), ('animals', 0.5972222089767456), ('facts', 0.6721044182777405), ('companies', 0.8066666722297668)]
# 0.7077413201332092 [('capitals', 0.8079560995101929), ('inventions', 0.6952054500579834), ('elements', 0.6236559152603149), ('animals', 0.636904776096344), ('facts', 0.6818923354148865), ('companies', 0.8008333444595337)]

###################### -4
#0.6588431100050608 [('capitals', 0.8319615721702576), ('inventions', 0.6073059439659119), ('elements', 0.5795698761940002), ('animals', 0.574404776096344), ('facts', 0.6623165011405945), ('companies', 0.6974999904632568)]
#0.6627328594525655 [('capitals', 0.7942386865615845), ('inventions', 0.551369845867157), ('elements', 0.6107527017593384), ('animals', 0.5773809552192688), ('facts', 0.6851549744606018), ('companies', 0.7574999928474426)]
#0.6305222113927206 [('capitals', 0.693415641784668), ('inventions', 0.5696346759796143), ('elements', 0.5817204117774963), ('animals', 0.5813491940498352), ('facts', 0.6378466486930847), ('companies', 0.7191666960716248)]
## 8/23/23: 0.6653042634328207 [('capitals', 0.7716049551963806), ('inventions', 0.6175799369812012), ('elements', 0.6000000238418579), ('animals', 0.5605158805847168), ('facts', 0.7096247673034668), ('companies', 0.7325000166893005)]


###########for rebuttle:
######## Random (for -12): 0.5067621270815531 [('capitals', 0.4855967164039612), ('inventions', 0.5273972749710083), ('elements', 0.5086021423339844), ('animals', 0.5158730149269104), ('facts', 0.5106036067008972), ('companies', 0.4925000071525574)]
######## same 0.4962475597858429 [('capitals', 0.49862825870513916), ('inventions', 0.5182648301124573), ('elements', 0.4817204177379608), ('animals', 0.4910714328289032), ('facts', 0.48613375425338745), ('companies', 0.5016666650772095)]
######### Random train: 0.6259963313738505 [('capitals', 0.6077371835708618), ('inventions', 0.6337108612060547), ('elements', 0.6473326683044434), ('animals', 0.6198542714118958), ('facts', 0.621345043182373), ('companies', 0.6259979605674744)]
######### SALMA train: 0.8633201519648234 [('capitals', 0.8154311776161194), ('inventions', 0.8750240206718445), ('elements', 0.8933074474334717), ('animals', 0.8800472617149353), ('facts', 0.8667762875556946), ('companies', 0.849334716796875)]
##########same: 0.8650420208772024 [('capitals', 0.8199697136878967), ('inventions', 0.8790555000305176), ('elements', 0.8855479955673218), ('animals', 0.8963955044746399), ('facts', 0.8605628609657288), ('companies', 0.8487205505371094)]
############ LR (for -12): 0.685535192489624 [('capitals', 0.7818930149078369), ('inventions', 0.7260273694992065), ('elements', 0.5989247560501099), ('animals', 0.613095223903656), ('facts', 0.6182708144187927), ('companies', 0.7749999761581421)]
######## LR 150M layer -8 0.5559209485848745[('capitals', 0.5466392040252686), ('inventions', 0.5593607425689697), ('elements', 0.5096774101257324), ('animals', 0.5565476417541504), ('facts', 0.5399673581123352), ('companies', 0.6233333349227905)]
#########LR 350M layer -4 0.5603739321231842 [('capitals', 0.5356652736663818), ('inventions', 0.594748854637146), ('elements', 0.5290322303771973), ('animals', 0.5307539701461792), ('facts', 0.559543251991272), ('companies', 0.612500011920929)]

#########FULL 125M layer -4 0.5518043637275696 [('capitals', 0.5850480198860168), ('inventions', 0.5627853870391846), ('elements', 0.5408602356910706), ('animals', 0.5178571343421936), ('facts', 0.5334420800209045), ('companies', 0.5708333253860474)]
#########FULL 125M layer -4 0.5676246285438538 [('capitals', 0.5843621492385864), ('inventions', 0.586758017539978), ('elements', 0.5333333611488342), ('animals', 0.5248016119003296), ('facts', 0.5464926362037659), ('companies', 0.6299999952316284)]
#########FULL 350M layer -4 0.5794083575407664 [('capitals', 0.6008230447769165), ('inventions', 0.6073059439659119), ('elements', 0.550537645816803), ('animals', 0.5059523582458496), ('facts', 0.569331169128418), ('companies', 0.6424999833106995)]
#########FULL 350M layer -4 0.5801965196927389 [('capitals', 0.6275720000267029), ('inventions', 0.6187214851379395), ('elements', 0.5623655915260315), ('animals', 0.52182537317276), ('facts', 0.5448613166809082), ('companies', 0.6058333516120911)]
#########FULL 350M layer -8 0.5700395603974661 [('capitals', 0.5651577711105347), ('inventions', 0.611872136592865), ('elements', 0.5591397881507874), ('animals', 0.5158730149269104), ('facts', 0.5448613166809082), ('companies', 0.6233333349227905)]
####### 1.3b -4 0.6220493117968241 [('capitals', 0.7427983283996582), ('inventions', 0.6061643958091736), ('elements', 0.5537634491920471), ('animals', 0.5188491940498352), ('facts', 0.6590538620948792), ('companies', 0.6516666412353516)]
########1.3b -8 0.6414273679256439 [('capitals', 0.7599451541900635), ('inventions', 0.6015982031822205), ('elements', 0.5784946084022522), ('animals', 0.5753968358039856), ('facts', 0.6247960925102234), ('companies', 0.7083333134651184)]
##### 2.7b -4 0.6367523968219757 [('capitals', 0.7112482786178589), ('inventions', 0.6267123222351074), ('elements', 0.5763440728187561), ('animals', 0.550595223903656), ('facts', 0.6639478206634521), ('companies', 0.6916666626930237)]
##### 2.7b -8 0.6765809257825216 [('capitals', 0.8237311244010925), ('inventions', 0.6312785148620605), ('elements', 0.5935483574867249), ('animals', 0.5694444179534912), ('facts', 0.6623165011405945), ('companies', 0.7791666388511658)]


#######LLAMA7
####only facts -4: 0.765040647983551 [(0, 0.7560975551605225), (1, 0.772357702255249), (2, 0.7479674816131592), (3, 0.7886179089546204), (4, 0.7642276287078857), (5, 0.7479674816131592), (6, 0.7479674816131592), (7, 0.7967479825019836), (8, 0.7642276287078857), (9, 0.7642276287078857)]
# -8: 0.7650406420230865 [(0, 0.7804877758026123), (1, 0.7398374080657959), (2, 0.7804877758026123), (3, 0.7642276287078857), (4, 0.7886179089546204), (5, 0.7804877758026123), (6, 0.7560975551605225), (7, 0.7642276287078857), (8, 0.7479674816131592), (9, 0.7479674816131592)]
# -1: 0.7463414669036865 [(0, 0.7235772609710693), (1, 0.7479674816131592), (2, 0.7154471278190613), (3, 0.7317073345184326), (4, 0.7317073345184326), (5, 0.7560975551605225), (6, 0.7479674816131592), (7, 0.8048780560493469), (8, 0.772357702255249), (9, 0.7317073345184326)]

#######LLAMA7 ALL
# -1 (x3): 0.7106903294722239 [('capitals', 0.7574302752812704), ('inventions', 0.6735159754753113), ('elements', 0.68136199315389), ('animals', 0.7337962985038757), ('facts', 0.7444263299306234), ('companies', 0.6736111044883728)]
# -4 (x3): 0.7320797178480359 [('capitals', 0.8145861824353536), ('inventions', 0.7207001447677612), ('elements', 0.6767025192578634), ('animals', 0.7248677412668864), ('facts', 0.7661772767702738), ('companies', 0.6894444425900778)]
# -8 (x3): 0.762224018573761 [('capitals', 0.8721993565559387), ('inventions', 0.7815829515457153), ('elements', 0.6849462389945984), ('animals', 0.7394179900487264), ('facts', 0.7857531309127808), ('companies', 0.7094444433848063)]
# -12 (x3): 0.8059949543741015 [('capitals', 0.882030189037323), ('inventions', 0.8458904027938843), ('elements', 0.6949820717175802), ('animals', 0.7757936517397562), ('facts', 0.8053289651870728), ('companies', 0.8319444457689921)]
# -16 (x3): 0.8297899001174504 [('capitals', 0.9222679336865743), ('inventions', 0.8938356041908264), ('elements', 0.6939068039258321), ('animals', 0.7774470845858256), ('facts', 0.8254486322402954), ('companies', 0.8658333420753479)]

