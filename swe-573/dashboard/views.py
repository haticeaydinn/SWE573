from django.shortcuts import render
from django.http import HttpResponse
from login.models import CustomUserModel
from matplotlib.figure import Figure
import io
import matplotlib.pyplot as plt; plt.rcdefaults()
from collections import Counter
import praw
import re
from praw.models import MoreComments
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords

# from django.contrib.auth.models import User
# from .forms import NameForm


# Create your views here.
def index(request):
    # users = User.objects.all()
    if request.user.is_authenticated:
        user_selected = CustomUserModel.objects.get(user_id=request.user.id)
        # user_name = request.user.username
        # users = auth_user.objects.get(username=user_name)
        # return HttpResponse("Hello SWE world !")
        data = request.POST.get('name')
        print(data)
        global val
        def val():
            return data
        # request.session['search_word'] = data
        date_interval = request.POST.get('date_interval')
        global val2
        def val2():
            return date_interval
        return render(request, 'index.html', {'data':data, 'date_interval':date_interval, 'user_selected': user_selected})


def mplimage(request):
    fig = Figure()

    reddit = praw.Reddit(
        client_id="cb6EqxN8HSVbXQ",
        client_secret="GSzNjdYdnpS4JxebAq5WHLeJuiwv8g",
        user_agent="my user agent"
    )

    # search_word = request.session['search_word']
    searched_word = val()
    selected_date = val2()
    print(searched_word)
    print(selected_date)

    content = []

    '''
    for comment in reddit.subreddit("COVID+Coronavirus+COVID19").comments(limit=100):
        content.append(comment.body)
    '''

    '''
    for submission in reddit.subreddit("all").search("covid", time_filter='day'):
        for top_level_comment in submission.comments[0:2]:
            if isinstance(top_level_comment, MoreComments):
                continue
            content.append(top_level_comment.body)
    '''
    
    for submission in reddit.subreddit("all").search(searched_word, time_filter=selected_date):
        # content.append(submission.selftext)
        content.append(submission.title)


    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f921"
                                   u"\u24b6"
                                   u"\u010d"
                                   u"\U0001f914"
                                   u"\u0103"
                                   u"\u0219"
                                   u"\u021b"
                                   u"\U0001f923"
                                   u"\U0001f92a"
                                   u"\U0001f92f"
                                   u"\U0001f90d"
                                   u"\U0001f970"
                                   u"\U0001f918"
                                   u"\U0001f974"
                                   u"\U0001f937"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    # with open("file.txt", 'w', encoding="mbcs") as filetowrite:
    '''
    with open("file.txt", 'w') as filetowrite:
        for row in content:
            s = "".join(map(str, remove_emoji(row)))
            filetowrite.write(s + '\n')

    '''
    import emoji

    with open("file.txt", 'w', encoding='utf-8') as filetowrite:
        for row in content:
            # s = "".join(map(str, remove_emoji(row)))
            s = emoji.get_emoji_regexp().sub(u'', row)
            filetowrite.write(s + '\n')



    # load data
    filename = 'file.txt'
    file = open(filename, 'rt', encoding='utf-8')
    text = file.read()
    file.close()
    # split into words
    
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if (not w in stop_words and not w in searched_word)]

    # word_freq = Counter(nouns)
    word_freq = Counter(words)
    common_nouns = word_freq.most_common(10)

    x_n = []
    for i in range(len(common_nouns)):
        word = common_nouns[i][0]
        x_n.append(word)

    y_n = []
    for i in range(len(common_nouns)):
        freq = common_nouns[i][1]
        y_n.append(freq)

    plt.switch_backend('agg')
    f, ax = plt.subplots(figsize=(9, 4))  # set the size that you'd like (width, height)
    # plt.bar(x_n, y_n)
    # deneme
  
    ax.bar(x_n, y_n,width=0.4)
    #Now the trick is here.
    #plt.text() , you need to give (x,y) location , where you want to put the numbers,
    #So here index will give you x pos and data+1 will provide a little gap in y axis.
    for index,data in enumerate(y_n):
        plt.text(x=index , y =data+0.2 , s=f"{data}" , fontdict=dict(fontsize=10))

    # deneme son
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Common Words')

    plt.savefig('example.png')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


def display_text(request):
    '''file1 = open('file.txt', 'r', encoding='utf-8')
    d = file1.read()
    return render(request,'displaypost.html',{'dat':d})
    '''
    file2 = open('file.txt', 'r', encoding='utf-8') 
    count = 0
    disp_list = ["Post Titles","\n","\n"]

    # Using for loop 
    for line in file2: 
        count += 1

        disp_list.append(count)
        disp_list.append("\t")
        disp_list.append(line)
        disp_list.append("\n")
    
    # Closing files 
    file2.close()
    return HttpResponse(disp_list, content_type="text/plain")



def tagme_result(request):
    import tagme
    # Set the authorization token for subsequent calls.
    tagme.GCUBE_TOKEN = "a5a377c1-1bd0-47b9-907a-75b1cdacb1d9-843339462"

    '''
    with open('file.txt', 'r') as f:
        first_line = f.readline()

    lunch_annotations = tagme.annotate(first_line)

    ann_list = []
    # Print annotations with a score higher than 0.1
    for ann in lunch_annotations.get_annotations(0.1):
        ann_list.append(ann)
    '''

    file1 = open('file.txt', 'r', encoding='utf-8') 
    count = 0
    ann_list = ["Entity Results","\n","\n"]

    # Using for loop 
    for line in file1: 
        count += 1
        lunch_annotations = tagme.annotate(line)
        ann_list.append(count)
        ann_list.append("\n")

        # Print annotations with a score higher than 0.1
        for ann in lunch_annotations.get_annotations(0.1):
            ann_list.append(ann)
            ann_list.append("\t")
            ann_list.append(ann.uri())
            ann_list.append("\n")
    
    # Closing files 
    file1.close() 

    return HttpResponse(ann_list, content_type="text/plain")


def sentiment_analysis(request):
    from textblob import TextBlob

    file2 = open('file.txt', 'r', encoding='utf-8') 
    count = 0
    sent_list = []

    # Using for loop 
    for line in file2: 
        count += 1
        analysis = TextBlob(line).sentiment

        sent_list.append(count)
        sent_list.append("\t")
        sent_list.append(analysis)
        sent_list.append("\n")
    
    # Closing files 
    file2.close()
    return HttpResponse(sent_list, content_type="text/plain")