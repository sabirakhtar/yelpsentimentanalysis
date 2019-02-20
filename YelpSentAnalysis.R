library(tm)
library(stringi)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(dplyr)
library(tidytext)
library(stringr)
library(reshape2)
library(ggplot2)
library(e1071)
library(caret)
library(glmnet)
library(RWeka)
library(slam)
library(jsonlite)
library(NLP)
library(plotly)
library(broom)
library(Matrix)
library(randomForest)
library(readr)
library(irlba)
library(RXKCD)

############ Reading Data file
YElp_reviews <- read_csv("YElp_reviews.csv",   
                         col_types = cols(X1 = col_skip()))

sample_reviews=sample_n(YElp_reviews,5000)

#DATA EXPLORATION PHASE
dim(sample_reviews)

sample_reviews=as.data.frame(sample_reviews)

sapply(sample_reviews,class)

sample_reviews$date=as.Date(sample_reviews$date,"%m/%d/%Y")

hist(sample_reviews$stars, xlab = "Star Rating", ylab="Count",col = "red") #######Distribution of stars using these stars to create the 
#######dependent variable for the first methid

rating <- aggregate(sample_reviews$stars,list(sample_reviews$stars),length)
names(rating)<-c("Stars","Total")

useful <- aggregate(sample_reviews$useful,list(sample_reviews$stars),sum)
names(useful)<-c("Stars","Useful Reviews")

funny <- aggregate(sample_reviews$funny,list(sample_reviews$stars),sum)
names(funny)<-c("Stars","Funny Reviews")

cool <- aggregate(sample_reviews$cool,list(sample_reviews$stars),sum)
names(cool)<-c("Stars","Cool Reviews")

review_rating_type <- merge(rating,useful,by= "Stars")
review_rating_type <- merge(review_rating_type,funny,by= "Stars")
review_rating_type <- merge(review_rating_type,cool,by= "Stars")

#Percentage
#par(mfrow=c(1,3))
plot(review_rating_type$Stars,(review_rating_type$`Useful Reviews`/review_rating_type$Total) *100, type = "b", xlab = "Stars", ylab ="% Useful Ratings", main = "Relationship between Star Reviews and Useful Ratings", col ="darkblue",lwd=1.5)

plot(review_rating_type$Stars,(review_rating_type$`Funny Reviews`/review_rating_type$Total)*100, type = "b", xlab = "Stars", ylab ="% Funny Ratings", main = "Relationship between Star Reviews and Funny Ratings", col ="red",lwd=1.5)

plot(review_rating_type$Stars,(review_rating_type$`Cool Reviews`/review_rating_type$Total)*100, type = "b", xlab = "Stars", ylab ="% Cool Ratings", main = "Relationship between Star Reviews and Cool Ratings", col ="orange",lwd=1.5)



#### tf-idf
#### Though we have already removed very common "stop words" from our analysis, 
#### it is good practice in quantitative text analysis to identify unusual words that might set one document apart from the others. 
#### The metric most commonly used to identify these unusual words is "Term Frequency Inverse Document 


##########Finding Unusual words ########
Unusual_Words<- sample_reviews %>%
dplyr::select(date,text) %>%
unnest_tokens("word", text) %>%
anti_join(stop_words) %>%
count(word, date) %>%
bind_tf_idf(word, date, n)


#Now let's see what the most unusual words are:
top_tfidf<-Unusual_Words %>%
arrange(desc(tf_idf))
top_tfidf$word[1:10]

#### The tfidf increases the more a term appears in a document but it is negatively weighted by the overall frequency of terms across all documents in the dataset or Corpus.
#### In simpler terms, the tf-idf helps us capture which words are not only important within a given document but also distinctive vis-a-vis the broader corpus or tidytext dataset.

##### Word Counting
Word_counting<- sample_reviews %>%
  select(date,text) %>%
  unnest_tokens("word", text)


### Dictionary Method 1
###Let's apply the bing sentiment dictionary to our database yelp reviews:

dictionary_bing<- Word_counting %>%
inner_join(get_sentiments("bing")) %>%
count(date, sentiment) 

head(dictionary_bing)

positive_sentiment =
  Word_counting %>%
  inner_join(get_sentiments("bing")) %>% 
  filter(sentiment=="positive") %>%
  count(date, sentiment)

negative_sentiment =
  Word_counting %>%
  inner_join(get_sentiments("bing")) %>% 
  filter(sentiment=="negative") %>%
  count(date, sentiment)

ggplot(negative_sentiment, aes(x=date, y=n))+
  geom_line(color="red")+
  theme_minimal()+
  ylab("Frequency of Negative Words in Yelp Reviews")+
  xlab("Date")

ggplot(positive_sentiment, aes(x=date, y=n))+
  geom_line(color="blue")+
  theme_minimal()+
  ylab("Frequency of Positive Words in Yelp Reviews")+
  xlab("Date")

##########Creating the dependent variable based on stars working on sample of the entire dataset##############

sample_reviews$Liked=ifelse(sample_reviews$stars>3,1,0)
head(sample_reviews)
table(sample_reviews$Liked)
sample_reviews=as.data.frame(sample_reviews)

#usableText=data.frame(str_replace_all(reviews$text,"[^[:graph:]]", " ")) 

sample_reviews <- sample_reviews[order(sample_reviews[,5]),]
# First we will delete non-recognizable characters, otherwise the tm package will get in trouble later. 
# Note that this can also delete some emoticons
reviewsText <- sapply(sample_reviews[,6],function(x) iconv(x, 'utf8', 'ascii',""))


corpus <- VCorpus(VectorSource(reviewsText))


#Cleaning reviews text

corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
#corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

#load("wordListSpelling.RData")

#Function to correct misspelled words
correct <- function(word) {
  # How dissimilar is this word from all words in the wordlist?
  edit_dist <- adist(word, wordlist)
  # Is there a word that reasonably similar?
  # If yes, which ones?
  # If no, append the original word to
  # Select the first result (because wordlist is sorted
  # from most common to least common)
  c(wordlist[edit_dist <= min(edit_dist,2)],word)[1]
}


######### Don't run  as it takes a lot of time to run for 5000 reviews
######### Execution time 1hr 46 mins to run spell check

# Note that the above function is for words, not for documents/sentences. 
# However, in the corpus, we look at documents
# So we have two option: 
#   - adapt the function above to work on sentences (e.g., by performing tokenization within)
#   - Work directly on the words. 
# Here, we will show the second approach, which means we will take words out of the corpus first

# 
# commentsUnlisted <- unlist(sapply(corpus,'[',"content"))
# comments_spell_checked <- character(length(commentsUnlisted))
# 
# for (i in 1: length(commentsUnlisted)){
#   words <- unlist(strsplit(commentsUnlisted[i],' '))
#   words <- as.character(sapply(words,correct))
# 
#    #Concatenate back to a string
#   comments_spell_checked[i] <-
#     paste(words, collapse=" ")
#  }

# comments <- VCorpus(VectorSource(comments_spell_checked))


################End of Preprocessing Document#######################
### Step 3:  Create the term-document matrix
######
######


# Make the document-term matrix. 

# Note: this required the NGramTokenizer from the Weka package, and this needs Java (64-bit) to be installed
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))

dtmComments <- DocumentTermMatrix(corpus, control = list( tokenizers = BigramTokenizer,
                                                          weighting =function(x) weightTf(x)))

dtmComments <- removeSparseTerms(dtmComments,0.99)

######## EDA of the CORPUS #################

# 1. Word cloud based on the original text
# use the termFreq of the tm package
# This also uses a tokenizer inside
tf <- termFreq(reviewsText)
wordcloud(names(tf),tf,
          max.words=50,
          scale=c(3,1))

# 2. Word cloud based on the dtm matrix and all the conversions

m <- as.matrix(dtmComments)
#Count occurrences of each term
v <- sort(colSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)

#create word cloud
options(warn=-1) #turn warnings off
wordcloud(d$word,d$freq,
          max.words=50,
          scale=c(3,1))

###WordCloud of Corpus
wordcloud(corpus, max.words = 50, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))


########## Top 20 words and their frequency

wordcount <- colSums(m)
top <- head(sort(wordcount, decreasing=TRUE), 20)

dfplot <- as.data.frame(melt(top))
dfplot$word <- dimnames(dfplot)[[1]]
dfplot$word <- factor(dfplot$word,
                      levels=dfplot$word[order(dfplot$value,
                                               decreasing=TRUE)])

fig <- ggplot(dfplot, aes(x=word, y=value)) + geom_bar(stat="identity")
fig <- fig + xlab("Word in Corpus")
fig <- fig + ylab("Count")
fig <- fig + theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(fig)

########### Converting DTM to Dataframe

dtm_df <-tidy(dtmComments)
sample_reviews$document <- seq(1:5000)
reviews_df <- merge(subset(sample_reviews, select=c("review_id","stars","document","business_id")),dtm_df,by="document")


##### Identifying Positive and Negative Terms
##########Identifying positive and negative words using star ratings as an indicator (average of ratings wherever the word occurs in the review)

word_avgstar <- aggregate(reviews_df$stars,list(reviews_df$term),mean)
plot_ly(x = word_avgstar$x, type = "histogram")

word_avgstar <- word_avgstar %>%arrange(x)

#Negative Words
head(word_avgstar,10)
nw <- data.frame(head(word_avgstar,10)[1], head(word_avgstar,10)[2], stringsAsFactors = FALSE)
nw$Group.1 <- factor(nw$Group.1, levels = unique(nw$Group.1)[order(nw$x, decreasing = TRUE)])

plot_ly(y=nw$x, x=nw$Group.1, type = "bar") %>%
  layout(yaxis=list(type='linear'),title="Top 10 Negative Words")

#Positive Words
tail(word_avgstar,10)
pw <- data.frame(tail(word_avgstar,10)[1], tail(word_avgstar,10)[2], stringsAsFactors = FALSE)
pw$Group.1 <- factor(pw$Group.1, levels = unique(pw$Group.1)[order(pw$x, decreasing = TRUE)])

plot_ly(y=pw$x, x=pw$Group.1, type = "bar") %>%
  layout(yaxis=list(type='linear'),title="Top 10 Positive Words")



############ Alternative approach to finding positive and negative words
review_words <- reviews_df %>%
  mutate_all(as.character)

review_words_counted <- review_words %>%
  dplyr::count(review_id, business_id, stars, term) %>%
  ungroup()

#review_words_counted

word_summaries <- review_words_counted %>%
  dplyr::group_by(term) %>%
  dplyr::summarize(businesses = n_distinct(business_id),
            reviews = n(),
            uses = sum(n),
            average_stars = mean(as.numeric(stars))) %>%
  ungroup()
#word_summaries

#Words that are present in atleast 100 documents and in more than 5 businesses
word_summaries_filtered <- word_summaries %>%
  filter(reviews >= 100, businesses >= 5)
#word_summaries_filtered

#Positive words -  from the filtered list
word_summaries_filtered %>%
  arrange(desc(average_stars))

#Negative words - from filtered list
word_summaries_filtered %>%
  arrange(average_stars)


#Positive and Negative words as per AFINN dictionary
AFINN <- sentiments %>%
  dplyr::filter(lexicon == "AFINN") %>%
  dplyr::select(term = word, afinn_score = score)

words_afinn <- word_summaries_filtered %>%
  inner_join(AFINN)

words_afinn
  plot_ly(y=words_afinn$afinn_score, x=words_afinn$term, type = "bar") %>%
  layout(yaxis=list(type='linear'),title="Words with Afinn Dictionary and Sentiment Score")



####### Visualizing the relationships and distribution
ggplot(words_afinn, aes(afinn_score, average_stars, group = afinn_score)) +
  geom_boxplot() +
  xlab("AFINN Sentiment Score of word") +
  ylab("Average Stars of Reviews with this Word")


####### The graph illustrates different kind of words (positive or negative) that occur frequently in reviews rated low to high - It can be observed that the density of positive words begin to increase for reviews higher than rating of 3 stars.
mid<-mean(words_afinn$afinn_score)
ggplot(words_afinn, aes(x=reviews, y=average_stars,color=afinn_score)) + 
  geom_point()+ 
  scale_color_gradient2(midpoint=mid, low="red",high="dark blue", space ="Lab" )+
  geom_text(aes(label=term ,hjust=0, vjust=0),size=5)


########################################
###  Dictionary-based lookup           # mETHOD 2
########################################

reviews_sentiment <- review_words %>%
  inner_join(AFINN, by = "term") 
#reviews_sentiment

#Average Sentiment score for each review using AFINN dictionary
reviews_score <- reviews_sentiment%>%
  dplyr::group_by(review_id, stars) %>%
  dplyr::summarize(sentiment = mean(afinn_score))

#reviews_score

reviews_score <- reviews_score %>%
  right_join(subset(sample_reviews, select=c("review_id","text","document")), by = "review_id")


#Classifying review as positive and negative based on sentiment score
reviews_score <- mutate(reviews_score,
                        sentiment_cat = ifelse(sentiment >= 2,"Pos","Neg"))

head(reviews_score)
reviews_score=as.data.frame(reviews_score)


########################################
###  Dictionary-based lookup           # mETHOD 3 
########################################
# Load dictionary 

#read in dictionary
dictionary <- read.csv("SentimentDictionary.csv")
# there are other dictionaries available as well, eg in the sentiment packages in R

#Let's have a look
head(dictionary)

#VALENCE: 1= sad, 9= happy
#AROUSAL: 1= calm, 9=excited
#DOMINANCE: 1=controlled, 9=in control

#Now recode all columns so that neutral equals 0, -5= negative, 5=positive
dictionary[,2:4] <- sapply(dictionary[,2:4],function(x) x-5)


# Determine sentiment
# Note that we do not really need a document-term matrix for this
# All we need to do is to split the string in multiple substrings. 
# Therefore, we will only use the function 'strsplit'

sentimentScore <- numeric(length(reviewsText))

for (i in 1:length(reviewsText)){
  
  text <- tolower(reviewsText)
  
  split <- strsplit(text[i],split=" ")[[1]] 
  
  # Find the negative words in the split
  neg_words <- match(split, c("not","no","nor"))
  
  # Get the position of the word following the negative word
  neg_words <- which(!is.na(neg_words)) + 1
  
  # Get the word from the position found
  neg_split <- split[neg_words]
  
  # Remove the neg_split from the actual split 
  # (because we have to change the scores of these words to negative)
  # Eg. -4 to 4 and 4 to -4
  split <- split[!split %in% c(neg_split,"not","no","nor")]
  
  # Get the scores from dictionary split
  m <- match(split, dictionary$Word)
  present <- !is.na(m)
  
  # Get the scores from dictionary neg_split
  m_neg <- match(neg_split, dictionary$Word)
  present_neg <- !is.na(m_neg)
  
  # Create a vector with the scores of split and neg_split
  # Make sure you change the sign of the scores from neg_split
  # Eg. -4 to 4 and 4 to -4
  wordvalences <- c(dictionary$VALENCE[m[present]],(-1*dictionary$VALENCE[m_neg[present_neg]]))
  
  #compute the mean valence of the entire comment
  sentimentScore[i] <- mean(wordvalences, na.rm=TRUE)
  
  #handle the case when none of the words is in the dictionary
  if (is.na(sentimentScore[i])) sentimentScore[i] <- 0 else sentimentScore[i] <- sentimentScore[i]
  
}

#Let's look at the result
head(sentimentScore)
mean(sentimentScore)
sd(sentimentScore)




############ Method 2 Using Machine Learning Models
# Set sample (stratified)
# Make our dependent variable dichotomous
y <- as.factor(sample_reviews[,"Liked"])

levels(y)

# Define proportion to be in training set 
p <- 0.5

# Define observations to be in training set
class1_train <- sample(which(y==as.integer(levels(y)[1])), floor(p*table(y)[1]),replace=FALSE)
class2_train <- sample(which(y==as.integer(levels(y)[2])), floor(p*table(y)[2]),replace=FALSE)

training_locations <- c(class1_train,class2_train) 

# Create the training and test set now
# Store them in a list for easiness

txt_l <- list()
txt_l[[2]] <- list()

txt_l[[1]]<- sample_reviews[sort(training_locations),6]
txt_l[[2]]<- sample_reviews[-sort(training_locations),6]


# Make our training and test set corpora
for (i in 1:2){
  txt_l[[i]] <- VCorpus(VectorSource((txt_l[[i]])))
}


# create a function that allows to make the training and test set correctly, with the n-grams specified
Ngram <- function(inputset1,inputset2,mindegree,maxdegree){
  # inputset1 = training dataset
  # inputset2 = test dataset
  # mindegree = minimum n-gram
  # maxdegree = maximum n-gram
  
  outputlist <- list()
  
  # training
  Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = mindegree, max = maxdegree))
  tr <- DocumentTermMatrix(inputset1, control = list(tokenize = Tokenizer,
                                                     weighting = function(x) weightTf(x),
                                                     RemoveNumbers=TRUE,
                                                     removePunctuation=TRUE,
                                                     stripWhitespace= TRUE))
  # test
  test <- DocumentTermMatrix(inputset2, control = list(tokenize = Tokenizer,
                                                       weighting = function(x) weightTf(x),
                                                       RemoveNumbers=TRUE,
                                                       removePunctuation=TRUE,
                                                       stripWhitespace= TRUE))
  
  # Apply sparseness reduction 
  # also reducing the number of documents (respondents) because there will be rows which will not have values anymore
  
  #tr <- removeSparseTerms(tr,0.9999)
  
  # Reform the test DTM to have the same terms as the training case 
  # Remember that, because we will use a model, that our test set should contain the same elements as our training dataset
  Intersect <- test[,intersect(colnames(test), colnames(tr))]
  diffCol <- tr[,setdiff(colnames(tr),colnames(test))]
  newCols <- as.simple_triplet_matrix(matrix(0,nrow=test$nrow,ncol=diffCol$ncol))
  newCols$dimnames <- diffCol$dimnames
  testNew<-cbind(Intersect,newCols)
  testNew<- testNew[,colnames(tr)]
  
  ## Convert term document matrices to common sparse matrices to apply efficient SVD algorithm
  
  dtm.to.sm <- function(dtm) {sparseMatrix(i=dtm$i, j=dtm$j, x=dtm$v,dims=c(dtm$nrow, dtm$ncol))}
  
  outputlist<- list(train=dtm.to.sm(tr),test=dtm.to.sm(testNew))
  
  return(outputlist)
}

# apply our function
# we store this in a new list, unigram

unigram <-Ngram(txt_l[[1]],txt_l[[2]],1,1)


# 4. Apply Singular Value Decomposition
# SVD will help to reduce this to a selected number of terms
# Note that we implemented an approximation with the package irlba, since the 'normal' svd gets stuck with very large datasets

SVD_all <- function(inputset,k){
  outputlist <- list()
  
  outputlist[[i]]<-list()
  
  trainer <- irlba(t(inputset[[1]]), nu=k, nv=k)
  tester <- as.data.frame(as.matrix(inputset[[2]] %*% trainer$u %*%  solve(diag(trainer$d))))
  
  outputlist<- list(train = as.data.frame(trainer$v), test= tester)
  
  return(outputlist)
}

svdUnigram <- SVD_all(unigram,70)


# 5. Prediction models

# Create datasets to use: append our dependent variable to our dataset 


train  <- cbind(y[sort(training_locations)],svdUnigram[[1]])
test <- cbind(y[-sort(training_locations)],svdUnigram[[2]])


## Apply Random Forest

RF_model_train <- randomForest(x=train[,2:dim(train)[[2]]],y=train[,1],importance=TRUE,ntree=1001)
RF_predict <- predict(RF_model_train,test[,2:dim(test)[[2]]],type = "prob")[,2]
# This returns the probabilities, which is more useful for the evaluation measures

RF_predict_resp <- predict(RF_model_train,test[,2:dim(test)[[2]]],type = "response")

# Calculate auc
if (!require("ROCR")) install.packages("ROCR", quiet=TRUE) ; require("ROCR")

predML <- prediction(RF_predict,test[,1])

# ROC curve
perfML <- performance(predML,"tpr","fpr")
plot(perfML)
abline(0,1)

## auc
auc.perfML = performance(predML, measure = "auc")
auc.perfML@y.values


conf.mat <- confusionMatrix(RF_predict_resp, test[,1])
conf.mat



#################### Method 2 Machine Learning Naive Bayes ########################

#Creating test and train corpus
revs.train <- corpus[1:3500]
revs.test <- corpus[3501:5000]

#Creating test and train dataset
df.train <- reviews_score[1:3500,]
df.test <- reviews_score[3501:5000,]




#### Keeping words that occur in atleast 15 reviews
#### Use most frequent words (fivefreq) to build the train and test DTM

fivefreq <- findFreqTerms(dtmComments, 15)
length((fivefreq))

wordcloud(fivefreq, max.words = 50, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

dtm.train <- DocumentTermMatrix(revs.train, control=list(dictionary = fivefreq))
dim(dtm.train)


dtm.test <- DocumentTermMatrix(revs.test, control=list(dictionary = fivefreq))
dim(dtm.test)


convert_count_to_boolean <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

trainNB <- apply(dtm.train, 2, convert_count_to_boolean)
testNB <- apply(dtm.test, 2, convert_count_to_boolean)


classifier <- naiveBayes(trainNB, as.factor(df.train$sentiment_cat), laplace = 0)

pred <- predict(classifier, newdata=testNB)
cm=table("Predictions"= pred,  "Actual" = as.factor(df.test$sentiment_cat))
Accuracy=(cm[1]+cm[4])/sum(cm)




########################Homberg Method#############################
############## We manually classified 350 reviews as positive and negative #################

cleaned_review <- read_csv("cleaned_review.csv")
cleaned_review=as.data.frame(cleaned_review)

table(cleaned_review$sentiment) ########Finding the number of positive and negative reviews


#Creating two dataset, one for negative reviews, and one for positive reviews
Positive <- subset(cleaned_review, sentiment == "1")
Negative <- subset(cleaned_review, sentiment == "0")


#Convert to corpus
corpusNeg= VCorpus(VectorSource(Negative$text))
corpusPos= VCorpus(VectorSource(Positive$text))


#Remove the less frequent words, and the useless ones for both dataset
corpusNeg = tm_map(corpusNeg, removeWords, stopwords())
dtmNeg <- TermDocumentMatrix(corpusNeg)
dtmNeg = removeSparseTerms(dtmNeg,0.99)
mNeg <- as.matrix(dtmNeg)
vNeg <- sort(rowSums(mNeg),decreasing=TRUE)
dfNeg <- data.frame(word = names(vNeg),freq=vNeg)

corpusPos = tm_map(corpusPos, removeWords, stopwords())
dtmPos <- TermDocumentMatrix(corpusPos)
dtmPos = removeSparseTerms(dtmPos,0.99)
mPos <- as.matrix(dtmPos)
vPos <- sort(rowSums(mPos),decreasing=TRUE)
dfPos <- data.frame(word = names(vPos),freq=vPos)


#After running summary statistics, we define the cutoff of occurence at 2 (=mean distribution)
#All the words that occured less than 3 times will be removed


dfNegTh <- subset.data.frame(dfNeg, dfNeg$freq > 2)
dfPosTh <- subset.data.frame(dfPos, dfPos$freq > 2)

########### WordCloud Positive Words ##############
options(warn=-1)
wordcloud(dfPos$word,dfPos$freq,
          max.words=40,
          scale=c(3,1))

Df_All <- merge(dfPosTh, dfNegTh,by="word")

#Write a function to get the relative score of occurence

score = function(dftotal,totalx,totaly){
  a=list()   
  b=list()
  score=list()
  for(i in 1:nrow(dftotal))
  {
    a[i]=dftotal$freq.x[i]/nrow(totalx)
    b[i]=dftotal$freq.y[i]/nrow(totaly)
    score[[i]]=(a[[i]]-b[[i]])/(a[[i]]+b[[i]])
  }
  return(score)
}

######### Finding Score for all the words#########
score_df=score(Df_All,Positive,Negative)


score_df=as.data.frame(unlist(score_df))
summary(score_df)
Positive=as.data.frame(Positive)
Negative=as.data.frame(Negative)

################# Creating corpus of all the words #################
Df_All$Absscore=score_df$`unlist(score_df)`
corp= VCorpus(VectorSource(Df_All$word))
dtm=DocumentTermMatrix(corp)


#Calculating the sentimet of the review based on the score calculated
for(i in 1:nrow(Positive)){
  
  split <- strsplit(Positive$text[i],split=" ")[[1]]
  
  m <- match(split, Df_All$word)
  
  present <- !is.na(m)
  
  wordAbsscore <- Df_All$Absscore[m[present]]
  
  Positive$sentiment[i] <- sum(wordAbsscore, na.rm=TRUE)
}

for(i in 1:nrow(Negative)){
  
  split <- strsplit(Negative$text[i],split=" ")[[1]]
  
  m <- match(split, Df_All$word)
  
  present <- !is.na(m)
  
  wordAbsscore <- Df_All$Absscore[m[present]]
  
  Negative$sentiment[i] <- sum(wordAbsscore, na.rm=TRUE)
}

finaldata=rbind(Positive,Negative)


#########Classifying the review as postive or negative based on the review sentiment
finaldata$Liked=ifelse(finaldata$sentiment>0,1,0)

table(finaldata$Liked)

finaldata$sentiment=NULL

############Final corpus of the dataset for ML############
corpusHomburg=VCorpus(VectorSource(finaldata$text))
dtmHomburg = DocumentTermMatrix(corpusHomburg)
dtmHomburg=removeSparseTerms(dtmHomburg,0.99)

# Make our dependent variable dichotomous
z <- as.factor(finaldata[,"Liked"])

levels(z)

# Define proportion to be in training set 
p <- 0.5

# Define observations to be in training set
class1_train <- sample(which(z==as.integer(levels(z)[1])), floor(p*table(z)[1]),replace=FALSE)
class2_train <- sample(which(z==as.integer(levels(z)[2])), floor(p*table(z)[2]),replace=FALSE)

training_locations <- c(class1_train,class2_train) 

# Create the training and test set now
# Store them in a list for easiness

txt_l <- list()
txt_l[[2]] <- list()

txt_l[[1]]<- finaldata[sort(training_locations),1]
txt_l[[2]]<- finaldata[-sort(training_locations),1]


# Make our training and test set corpora
for (i in 1:2){
  txt_l[[i]] <- VCorpus(VectorSource((txt_l[[i]])))
}


# create a function that allows to make the training and test set correctly, with the n-grams specified
Ngram <- function(inputset1,inputset2,mindegree,maxdegree){
  # inputset1 = training dataset
  # inputset2 = test dataset
  # mindegree = minimum n-gram
  # maxdegree = maximum n-gram
  
  outputlist <- list()
  
  # training
  Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = mindegree, max = maxdegree))
  tr <- DocumentTermMatrix(inputset1, control = list(tokenize = Tokenizer,
                                                     weighting = function(x) weightTf(x),
                                                     RemoveNumbers=TRUE,
                                                     removePunctuation=TRUE,
                                                     stripWhitespace= TRUE))
  # test
  test <- DocumentTermMatrix(inputset2, control = list(tokenize = Tokenizer,
                                                       weighting = function(x) weightTf(x),
                                                       RemoveNumbers=TRUE,
                                                       removePunctuation=TRUE,
                                                       stripWhitespace= TRUE))
  
  # Apply sparseness reduction 
  # also reducing the number of documents (respondents) because there will be rows which will not have values anymore
  
  #tr <- removeSparseTerms(tr,0.9999)
  
  # Reform the test DTM to have the same terms as the training case 
  # Remember that, because we will use a model, that our test set should contain the same elements as our training dataset
  Intersect <- test[,intersect(colnames(test), colnames(tr))]
  diffCol <- tr[,setdiff(colnames(tr),colnames(test))]
  newCols <- as.simple_triplet_matrix(matrix(0,nrow=test$nrow,ncol=diffCol$ncol))
  newCols$dimnames <- diffCol$dimnames
  testNew<-cbind(Intersect,newCols)
  testNew<- testNew[,colnames(tr)]
  
  ## Convert term document matrices to common sparse matrices to apply efficient SVD algorithm
  
  dtm.to.sm <- function(dtm) {sparseMatrix(i=dtm$i, j=dtm$j, x=dtm$v,dims=c(dtm$nrow, dtm$ncol))}
  
  outputlist<- list(train=dtm.to.sm(tr),test=dtm.to.sm(testNew))
  
  return(outputlist)
}

# apply our function
# we store this in a new list, unigram

unigram <-Ngram(txt_l[[1]],txt_l[[2]],1,1)


# 4. Apply Singular Value Decomposition
# In this case, not really necessary since we do not have a lot of terms
# however, in a normal situation, we would have lots of terms. 
# SVD will help to reduce this to a selected number of terms
# Note that we implemented an approximation with the package irlba, since the 'normal' svd gets stuck with very large datasets


if (!require("irlba")) install.packages("irlba", quiet=TRUE) ; require("irlba")

SVD_all <- function(inputset,k){
  outputlist <- list()
  
  outputlist[[i]]<-list()
  
  trainer <- irlba(t(inputset[[1]]), nu=k, nv=k)
  tester <- as.data.frame(as.matrix(inputset[[2]] %*% trainer$u %*%  solve(diag(trainer$d))))
  
  outputlist<- list(train = as.data.frame(trainer$v), test= tester)
  
  return(outputlist)
}

svdUnigram <- SVD_all(unigram,50)


# 5. Prediction models

# Create datasets to use: append our dependent variable to our dataset 

train  <- cbind(z[sort(training_locations)],svdUnigram[[1]])
test <- cbind(z[-sort(training_locations)],svdUnigram[[2]])


## SVM
classifier = svm(x=train[,2:dim(train)[[2]]],
                 y=train[,1],
                 type = 'C-classification',
                 kernel = 'linear')

y_pred = predict(classifier, test[,2:dim(test)[[2]]],type="prob")
y_pred_resp <- predict(classifier,test[,2:dim(test)[[2]]],type = "response")


# Calculate auc
if (!require("ROCR")) install.packages("ROCR", quiet=TRUE) ; require("ROCR")

predML <- prediction(as.numeric(y_pred),test[,1])

# ROC curve
perfML <- performance(predML,"tpr","fpr")
plot(perfML)
abline(0,1)

## auc
auc.perfML = performance(predML, measure = "auc")
auc.perfML@y.values


conf.mat <- confusionMatrix(y_pred_resp, test[,1])
conf.mat




