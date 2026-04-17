{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMtltF3v1yhwkUDbW5pNcJf",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/jeeva-anand/sentiment_analysis/blob/main/sentiment_analysis_.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "import nltk\n",
        "import string\n",
        "import warnings\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as pt\n",
        "\n",
        "pd.set_option(\"display.max_colwidth\",200)\n",
        "\n",
        "pd.set_option(\"display.max_row\",200)"
      ],
      "metadata": {
        "id": "sViCFd1KiBZO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7kMcJfTb1U7i"
      },
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Paste the path you copied from the file browser\n",
        "test = '/content/drive/MyDrive/Colab Notebooks/Sentiment Analysis/test_tweets_anuFYb8.csv'\n",
        "train = '//content/drive/MyDrive/Colab Notebooks/Sentiment Analysis/train_E6oV3lV.csv'\n",
        "\n",
        "test = pd.read_csv(test)\n",
        "train = pd.read_csv(train)\n",
        "\n",
        "train.head()\n",
        "\n"
      ],
      "metadata": {
        "id": "nuMimye6X_Uu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "os.listdir('/content/drive/MyDrive/Colab Notebooks/Sentiment Analysis/')"
      ],
      "metadata": {
        "id": "IXIw3j8fbG6U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train.info()"
      ],
      "metadata": {
        "id": "WgC-ELt5b3DS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train.describe()\n",
        "\n"
      ],
      "metadata": {
        "id": "qd_jeB8Kb732"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train.describe(include='object')"
      ],
      "metadata": {
        "id": "EPxpI8nrcEL0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train[train['label']==0].head(10)"
      ],
      "metadata": {
        "id": "wzS0CQuObqmi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train.shape"
      ],
      "metadata": {
        "id": "_B4V26VpcnAG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train['label'].value_counts()"
      ],
      "metadata": {
        "id": "9jhjGjlQcwaK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb = pd.concat([train,test],ignore_index=True)\n",
        "\n",
        "comb.tail(20)\n",
        "\n"
      ],
      "metadata": {
        "id": "S4eK2AWXeGQ1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def remove_pattern(pattern,text):\n",
        "   return re.sub(pattern,'',text)\n",
        "\n",
        "\n",
        "# print(remove_pattern('@[\\w]*',\"i/'m ML engineer @google & @Apple\"))"
      ],
      "metadata": {
        "id": "2dAU7kqphOI0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb['new_tweet']= np.vectorize(remove_pattern)(\"@\\w+\",comb['tweet'])\n",
        "\n",
        "comb.head()"
      ],
      "metadata": {
        "id": "DuiZxrSEmoeD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb['new_tweet']= np.vectorize(remove_pattern)(\"@[\\w+]*\",comb['new_tweet'])\n",
        "comb.head(20)"
      ],
      "metadata": {
        "id": "AW3JHOomTb3t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb.head()"
      ],
      "metadata": {
        "id": "IeuNDs9SThXg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# comb['tweet'].str.replace(\"#\\w+\",\"\",regex=True)\n",
        "# comb['tweet'].str.replace(\"@\\w+\",\"\",regex=True)\n",
        "# comb.head()"
      ],
      "metadata": {
        "id": "whaykJuYTsPr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# comb['new_tweet'] = comb['tweet']\n",
        "\n",
        "comb.head()"
      ],
      "metadata": {
        "id": "FoVd1L8Zn9Rr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb['new_tweet'] = comb['new_tweet'].str.replace(r\"[^a-zA-Z#]\",\" \",regex=True)\n",
        "\n",
        "comb.head()"
      ],
      "metadata": {
        "id": "Xi8vk_0bh6qn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# comb['new_tweet'] = comb['tweet']\n",
        "comb['new_tweet'] = comb['new_tweet'].str.replace(r'@[\\w]+',\" \",regex=True)\n",
        "\n",
        "comb.head()"
      ],
      "metadata": {
        "id": "eDCpHdMhi8kj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb['new_tweet'] = comb['new_tweet'].apply(lambda x: ' '.join([ w for w in x.split() if len(w) > 3]))\n",
        "\n",
        "comb.head()"
      ],
      "metadata": {
        "id": "pdSCJhqyjwTu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb['new_tweet'] = comb['new_tweet'].str.replace(\"[^a-zA-Z#]\",\" \",regex=True)\n",
        "\n",
        "comb.head()"
      ],
      "metadata": {
        "id": "A0NPw_Tio5lh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import nltk\n",
        "nltk.download('wordnet')\n",
        "\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "\n",
        "# ps = PorterStemmer()\n",
        "\n",
        "lm = WordNetLemmatizer()\n",
        "\n",
        "lm.lemmatize(\"singing\")"
      ],
      "metadata": {
        "id": "UmW1WFFkpOUF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "nltk.download('wordnet')\n",
        "nltk.download('omw-1.4')\n",
        "# print(nltk.data.path)\n",
        "\n",
        "from nltk.corpus import wordnet as wn\n",
        "\n",
        "wn.synsets(\"running\")\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "pAPbgqBJrOSc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "res = wn.synsets(\"running\")[1]\n",
        "\n",
        "# res.name()\n",
        "\n",
        "# res.definition()\n",
        "\n",
        "# res.examples()\n",
        "\n",
        "# lemm = res.lemmas()[1]\n",
        "\n",
        "# lemm.name()\n",
        "\n",
        "lemm = res.lemma_names()\n",
        "\n",
        "t1 = res.hypernyms()[0]\n",
        "\n",
        "t1.definition()\n",
        "\n",
        "lemm = res.lemmas()[0].antonyms()\n",
        "\n",
        "lemm\n"
      ],
      "metadata": {
        "id": "Hn_5fS6vxOKG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tokenize_tweet = comb['new_tweet'].apply(lambda x: x.split())\n",
        "\n",
        "tokenize_tweet.head()"
      ],
      "metadata": {
        "id": "9VyTYmDR1MOa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tokenize_tweet = comb['new_tweet'].str.split(\" \")\n",
        "\n",
        "tokenize_tweet.head()"
      ],
      "metadata": {
        "id": "ywspzVmA09QL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk import PorterStemmer\n",
        "\n",
        "ps = PorterStemmer()\n",
        "\n",
        "tokenize_tweet = tokenize_tweet.apply(lambda x: ([ps.stem(w) for w in x]))\n",
        "\n",
        "tokenize_tweet.head()"
      ],
      "metadata": {
        "id": "7knCwGlF017t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "for i in range(len(tokenize_tweet)):\n",
        "  tokenize_tweet[i] = \" \".join(tokenize_tweet[i])\n",
        "\n",
        "tokenize_tweet.head()"
      ],
      "metadata": {
        "id": "0fcybwNh3fU5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb['new_tweet'] = tokenize_tweet\n"
      ],
      "metadata": {
        "id": "4pLFlDeQ9XaS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "all_words = ' '.join(w for w in comb['new_tweet'])\n",
        "\n",
        "all_words"
      ],
      "metadata": {
        "id": "cotb2S1NJR9T"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from wordcloud import WordCloud\n",
        "\n",
        "all_words = ' '.join(w for w in comb['new_tweet'])\n",
        "\n",
        "print(all_words)\n",
        "\n",
        "wc = WordCloud(width=800, height= 800, random_state=21, max_font_size=110).generate(all_words)\n",
        "\n",
        "pt.figure(figsize=(10,7))\n",
        "pt.axis('off')\n",
        "pt.imshow(wc,interpolation=\"lanczos\")\n",
        "\n",
        "# pt.show()"
      ],
      "metadata": {
        "id": "mOPX3oky8KsD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "\n",
        "normal_words = ''.join([w for w in  comb['new_tweet'][comb['label'] == 0]])\n",
        "\n",
        "normal_words\n",
        "wc = WordCloud(width=800,height=500,random_state=21,max_font_size=100).generate(normal_words)\n",
        "pt.imshow(wc,interpolation=\"lanczos\")\n",
        "pt.axis('off')"
      ],
      "metadata": {
        "id": "16w32DqfPf1A"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "negative_words = ''.join([w for w in comb['new_tweet'][comb['label'] == 1]])\n",
        "\n",
        "wc = WordCloud(width=800,height=500,random_state=21,max_font_size=100).generate(negative_words)\n",
        "pt.imshow(wc,interpolation=\"lanczos\")\n",
        "pt.axis('off')"
      ],
      "metadata": {
        "id": "dnlaVQMSdsgy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def extract_hastag(s):\n",
        "  hashtag_bin = []\n",
        "  for i in range(len(s)):\n",
        "    hashtag_bin.append(re.findall(r'#[/w]+',i))\n",
        "  return hashtag_bin"
      ],
      "metadata": {
        "id": "LzDXeM9ugu-I"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train.head()"
      ],
      "metadata": {
        "id": "T9QwsiHRKiHT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# HT_normal = extract_hashtage(comb['new_tweet'][comb['label'] == 0])\n",
        "# HT_normal = extract_hashtag(comb.loc[comb['label'] == 0,'tweet'])\n",
        "\n",
        "# HT_normal\n"
      ],
      "metadata": {
        "id": "2j5ix_DmkIje"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def extract_hashtag(s):\n",
        "  hashtag_bin = []\n",
        "  for i in s:\n",
        "    hashtag_bin.append(re.findall(r'#[\\w]+',i))\n",
        "  return hashtag_bin\n",
        "\n",
        "# HT_normal = extract_hashtag(comb['new_tweet'][comb['label'] == 0])\n",
        "\n",
        "HT_normal = extract_hashtag(comb.loc[comb['label'] == 0,'new_tweet'])\n",
        "\n",
        "\n",
        "\n",
        "HT_negative = comb.loc[comb['label'] == 1,\"new_tweet\"].str.findall('#[\\w]+')\n",
        "\n",
        "# comb.head()\n",
        "\n",
        "HT_negative\n"
      ],
      "metadata": {
        "id": "6azwU9FKmQka"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ht_normal = sum(HT_normal,[])\n",
        "ht_neg = sum(HT_negative,[])\n",
        "\n",
        "ht_normal"
      ],
      "metadata": {
        "id": "nzXHegRomSH4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import seaborn as sns\n",
        "\n",
        "a = nltk.FreqDist(ht_normal)\n",
        "\n",
        "freq_ls = pd.DataFrame({'Hashtag':list(a.keys()),'Count':list(a.values())})\n",
        "\n",
        "freq_ls = freq_ls.nlargest(columns='Count',n=20)\n",
        "\n",
        "pt.figure(figsize=(16,5))\n",
        "sns.barplot(data=freq_ls, x = 'Hashtag', y = 'Count', palette='viridis')\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "kklvtv7ArqSl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
        "\n",
        "texts = [\"I am a SDE at google\",\"I was in dubai\"]\n",
        "\n",
        "vc = CountVectorizer()\n",
        "\n",
        "fit = vc.fit(texts)\n",
        "tran = vc. transform(texts)\n",
        "\n",
        "# print(vc.vocabulary_)\n",
        "print(vc.get_feature_names_out())\n",
        "\n",
        "print(tran.toarray())\n",
        "\n",
        "# print(X.toarray())"
      ],
      "metadata": {
        "id": "Amsv_YOwtMH2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "vc = CountVectorizer(max_df = 0.8, min_df =2, max_features = 10000, stop_words=\"english\",)\n",
        "\n",
        "X = vc.fit_transform(comb['new_tweet'])\n",
        "\n",
        "print(X.toarray())"
      ],
      "metadata": {
        "id": "bQDBOXbw4CmT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comb.shape"
      ],
      "metadata": {
        "id": "43DC7pnVKX18"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "'''\n",
        "1. TF - like cuntvector\n",
        "2. IDF - find important words\n",
        "3. TF*IDF - better than countvector\n",
        "\n",
        "1. TF\n",
        "\n",
        "[am, sde, at, google, was, in, dubai]\n",
        "\n",
        "s1: compute TF\n",
        "\n",
        "\"I am a SDE at google\"\n",
        "D1 = 1 1 1 1 0 0 0\n",
        "\n",
        "\"I was in dubai\"\n",
        "D2 = 0 0 0 0 1 1 1\n",
        "\n",
        "s2:compute IDF\n",
        "\n",
        "document feq\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "'''"
      ],
      "metadata": {
        "id": "piOEtSefLmdf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "texts = [\"I am a SDE at google\",\"I was in dubai\"]\n",
        "\n",
        "tf = TfidfVectorizer()\n",
        "\n",
        "\n",
        "tf.fit(texts)\n",
        "\n",
        "print(tf.get_feature_names_out())\n",
        "print(tf.idf_) # fit\n",
        "X = tf.transform(texts)\n",
        "\n",
        "print(X.toarray())\n",
        "\n"
      ],
      "metadata": {
        "id": "30VzOulMDPOV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "tf = TfidfVectorizer(min_df=1, max_df=0.9, max_features=1000, stop_words='english')\n",
        "\n",
        "X = tf.fit_transform(comb['new_tweet'])\n",
        "\n",
        "# print(X.toarray)\n",
        "\n",
        "X.shape"
      ],
      "metadata": {
        "id": "92AkMzf1J4jf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# Day 05\n",
        "\n"
      ],
      "metadata": {
        "id": "EBQe87eeKLsa"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}