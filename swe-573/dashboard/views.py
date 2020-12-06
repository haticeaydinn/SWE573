from django.shortcuts import render
from django.http import HttpResponse
from login.models import User
from matplotlib.figure import Figure
import io
import matplotlib.pyplot as plt; plt.rcdefaults()
import spacy
from collections import Counter
import praw
import re


# Create your views here.
def index(request):
    users = User.objects.all()
    # return HttpResponse("Hello SWE world !")
    return render(request, 'index.html', {'users': users})


def mplimage(request):
    fig = Figure()

    reddit = praw.Reddit(
        client_id="cb6EqxN8HSVbXQ",
        client_secret="GSzNjdYdnpS4JxebAq5WHLeJuiwv8g",
        user_agent="my user agent"
    )

    content = []

    for submission in reddit.subreddit("COVID").new(limit=100):
        content.append(submission.selftext)

    nlp = spacy.load('en')

    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    with open("C:/Users/ahmet/Desktop/SWE/SWE Fall 2020/SWE573 Suzan Hoca/Jupyter denemeleri/file.txt", 'w') as file:
        for row in content:
            s = "".join(map(str, remove_emoji(row)))
            file.write(s + '\n')

    docx = nlp(open('C:/Users/ahmet/Desktop/SWE/SWE Fall 2020/SWE573 Suzan Hoca/Jupyter denemeleri/file.txt').read())

    nouns = [token.text for token in docx if token.is_stop != True and token.is_punct != True
             and token.pos_ == 'NOUN']

    word_freq = Counter(nouns)
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
    plt.bar(x_n, y_n)
    plt.xlabel('Nouns')
    plt.ylabel('Frequency')
    plt.title('Most Common Nouns')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response
