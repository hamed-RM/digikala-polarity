from builtins import map
import tkinter as tk
from cProfile import label
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from pandas.core.series import Series
import regex as reg
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def tests():
    for i in range (2,66):
        if i==3:
            continue
        test=pd.read_excel (r'Answer\split_'+str(i)+'.xlsx')
        for j in range(0, test['label'].size):
            if (test['label'][j] != -1 and test['label'][j] != 0 and test['label'][j] != 1):
                print(i, 'in row:', j)



def remove_stopwords(string):
    temp = pd.read_excel(r'persian-stopwords-master\stop words.xlsx')
    my_stops_set=set(temp["stops"][:])
    words=reg.split(string=string,pattern=" ")
    words = [reg.sub(pattern='[a-zA-Z,0-9,.,!,?,(,),<,>,\n,\u200c,۰-۹]', repl=' ', string=i) for i in words]
    while ' ' in words:
        words.remove(' ')
    # print(my_stops_set)
    words = [w for w in words if not w in my_stops_set]
    # print(words)
    return (" ".join(words))
def click():
    id=int(txt.get())
    count_pos=0
    count_neu=0
    count_neg=0
    count_all=0
    df = pd.read_excel(r'output.xlsx')
    number = df['label'].size
    if var1.get()==1:
        for i in range(0,number):
            if(df['product_id'][i]==id):
                count_all+=1
                if(df['label'][i]==-1):
                    count_neg+=1
                elif (df['label'][i]==0):
                    count_neu+=1
                else:
                    count_pos+=1
            if(i==number-1 and count_all==0):
                print("there is no product with this id")
                return
    else:
        for i in range(0, number):
            if (df['user_id'][i] == id):
                count_all += 1
                if (df['label'][i] == -1):
                    count_neg += 1
                elif (df['label'][i] == 0):
                    count_neu += 1
                else:
                    count_pos += 1
            if (i == number - 1 and count_all == 0):
                print("there is no user with this id")
                return
    text=str("negetive: "+str((count_neg/count_all)*100)+"%"+\
                   "\n neutral: "+str((count_neu/count_all)*100)+"%"+\
                   "\n posetive: "+str((count_pos/count_all)*100)+"%"+
                    "\n comment counts: "+str(count_all))
    rs_lbl['text']=text

def graphic():
    window.geometry('500x103')
    window.title("Welcome to LikeGeeks app")
    lbl.place(x=0,y=20)
    txt.place(x=0,y=40)
    btn.place(x=0,y=70)
    rs_lbl.place(x=220,y=0)
    check_btn1.place(x=0,y=0)
    check_btn2.place(x=80,y=0)
    window.lift()
    window.mainloop()
def change_check_btn_state1():
    var1.set(1)
    var2.set(0)
def change_check_btn_state2():
    var2.set(1)
    var1.set(0)


data= pd.read_excel (r'Answer\split_'+str(2)+'.xlsx')
# tests()
temp=pd.DataFrame(data['label'])
s=time.time()
clean_train_reviews = []
print("cleaning train data")
for i in range(2,60):
    if(i==3):
        continue
    # print("cleaning", i)
    df= pd.read_excel (r'Answer\split_'+str(i)+'.xlsx')
    if i != 2:
        temp1 = pd.DataFrame(df['label'])
        temp=temp.append(temp1,ignore_index = True)
    num_reviews = df["comment"].size
    for i in range( 0, num_reviews):
        clean_train_reviews.append( remove_stopwords( str(df["comment"][i] )))

print("feature extraction started")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
print("vector",vectorizer)
print("train data feature",train_data_features)
# vocab = vectorizer.get_feature_names()
# print("vocab",vocab)
from sklearn.ensemble import RandomForestClassifier
print("create classifier")
forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data_features,temp['label'])

y_test_lable = []
y_test_user=[]
y_test_product=[]
clean_test_reviews = []
print("cleaning test data")
for j in range (60,66):
    test = pd.read_excel(r'Answer\split_'+str(j)+'.xlsx')
    # print("Cleaning test:", j)

    num_reviews = test["comment"].size
    for i in range(0,num_reviews):
        clean_review = remove_stopwords( str(test["comment"][i]) )
        clean_test_reviews.append( clean_review )
        y_test_lable.append(int(test['label'][i]))
        y_test_product.append(int(test['product_id'][i]))
        y_test_user.append(int(test['user_id'][i]))

y_test_lable=numpy.array(y_test_lable)
print("predicting:")
print("vec",vectorizer)
test_data_features = vectorizer.transform(clean_test_reviews)
print("test feature",test_data_features)
result = forest.predict(test_data_features)

ids={'product_id':y_test_product,
     'user_id':y_test_user,
     'label':result.tolist()}
print("save data in output.xlsx")
output=pd.DataFrame(ids,columns=['product_id','user_id','label'])
writer=pd.ExcelWriter('output.xlsx')
output.to_excel(writer)
writer.save()

print("accuracy: ",accuracy_score(y_test_lable,result))

showdata={'y_Actual':y_test_lable.tolist(),
          'y_Predicted':result.tolist()}
dataset=pd.DataFrame(showdata, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(dataset['y_Actual'], dataset['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()

e=time.time()
print("time:",e-s)
window = tk.Tk()
btn=tk.Button(window,text='Show',command=click,width=23,font=20)
lbl=tk.Label(window,text='Enter id:',font=20)
txt=tk.Entry(window,width=24,font=20)
rs_lbl=tk.Label(window,text="result will be shown here",
                            width=40,height=7,bg="yellow",anchor='c',wraplength=280
                            ,justify='center')
var1=tk.IntVar(value=1)
var2=tk.IntVar(value=0)
check_btn1=tk.Checkbutton(window,text='product id',command=change_check_btn_state1,variable=var1)
check_btn2=tk.Checkbutton(window,text='user id',command=change_check_btn_state2,variable=var2)
graphic()