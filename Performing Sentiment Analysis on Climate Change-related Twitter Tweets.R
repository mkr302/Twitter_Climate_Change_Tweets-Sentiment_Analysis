###################################################################################################################################################
#### File Name:      Performing Sentiment Analysis on Climate Change-related Twitter Tweets.R
#
#### Objective:      In this section, we will conduct sentiment analysis on Twitter data focusing on tweets related to climate
#                    change. By analyzing the sentiment expressed in these tweets, we aim to gain insights into public attitudes,
#                    opinions, and emotions regarding climate change. This analysis will provide valuable information on the
#                    overall sentiment and perception surrounding this important global issue.
#
#### Author:         Muralikrishnan Rajendran
#### Date:           20-Jul-2023
#### File Version:   v1.0
###################################################################################################################################################



### This code file has the following dependencies -
### 1) R version: 4.1.2 (2021-11-01) and above
### 2) Installing the required libraries (as mentioned below)
### 3) Following Source files placed in the working directory of R environment. The file is publicly available in Kaggle:
###     ---- twitter_sentiment_data.csv; Location: [Kaggle] https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset 
### 

#######################################################################################
### ****************************** Step 1: Pre-Steps **********************************
#######################################################################################
# Install the following required libraries (if not installed already in your local R env)

# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("lubridate")
# install.packages("reshape2")
# install.packages("tidyverse")
# install.packages("tidytext")
# install.packages("sentimentr")
# install.packages("stringr")
# install.packages("wordcloud")
# install.packages("textplot")
# install.packages("gridExtra")
# install.packages("patchwork")
# install.packages("tm")
# install.packages("stringr")

# Load the required libraries
suppressWarnings(suppressMessages(library(ggplot2)))
suppressWarnings(suppressMessages(library(dplyr))) # for %>%
suppressWarnings(suppressMessages(library(lubridate)))
suppressWarnings(suppressMessages(library(reshape2)))
suppressWarnings(suppressMessages(library(tidyverse)))
suppressWarnings(suppressMessages(library(tidytext)))
suppressWarnings(suppressMessages(library(sentimentr)))
suppressWarnings(suppressMessages(library(stringr)))
suppressWarnings(suppressMessages(library(wordcloud)))
suppressWarnings(suppressMessages(library(textplot)))
suppressWarnings(suppressMessages(library(gridExtra)))
suppressWarnings(suppressMessages(library(patchwork)))
suppressWarnings(suppressMessages(library(tm)))
suppressWarnings(suppressMessages(library(stringr)))



#######################################################################################
### Step 2: Loading the Twitter sentiment dataset, Data cleansing & EDA
#######################################################################################

# Read the Twitter dataset into a data frame
###########
## Data file: twitter_sentiment_data.csv
## Source: [Kaggle] https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset 
##
## The provided data source is a Twitter dataset related to climate change sentiment. It is stored in a CSV file named "twitter_sentiment_data.csv." The dataset is available on Kaggle.
## The dataset contains tweets related to climate change, and each tweet is labeled with a sentiment score. The sentiment score represents the attitude or belief expressed in the tweet 
## regarding man-made climate change. The possible sentiment labels are as follows:
## sentiment legend
## 2(News): the tweet links to factual news about climate change
## 1(Pro): the tweet supports the belief of man-made climate change
## 0(Neutral: the tweet neither supports nor refutes the belief of man-made climate change
## -1(Anti): the tweet does not believe in man-made climate change
##
## Researchers or data analysts can use this dataset to perform various sentiment analysis and natural language processing tasks related to climate change sentiment on Twitter. 
## They can investigate the prevalence of different sentiments, analyze trending topics, or study public opinions regarding climate change based on the content of the tweets in the dataset.
###########

## Load the Twitter sentiment dataset
twitter_data <- read.csv("twitter_sentiment_data.csv")

# Summary Statistics
summary(twitter_data)

# row count
nrow(twitter_data) 


####### Data Cleansing and Feature Engineering

# Tweets extracted from social media have unicode characters in them, which might affect our NLP processing (Sentiment analysis), hence removing unicode values twitter_data$messages is required, as shown below - 
remove_unicode <- function(text) {
  gsub("[^ -~]", "", text)
}

# Apply the remove_unicode function to the "message" column
twitter_data$message <- sapply(twitter_data$message, remove_unicode)

head(twitter_data, 10)

# datatype of columns
str(twitter_data)

# Checking for null values in columns
which(is.na(twitter_data$sentiment)) 
which(is.na(twitter_data$message)) 
which(is.na(twitter_data$tweetid)) 


# Check for null values in each column
null_columns <- colSums(is.na(twitter_data))

# Display columns with null values
cols_with_null <- names(null_columns[null_columns > 0])
cols_with_null 

# there are no columns with null values in the twitter_data data frame.


# Bar-plot on twitter_data$sentiment

# Order the levels of the sentiment column
twitter_data$sentiment <- factor(twitter_data$sentiment, levels = c(2, 1, 0, -1))

sum(twitter_data$sentiment == 2) 
sum(twitter_data$sentiment == 1) 
sum(twitter_data$sentiment == 0) 
sum(twitter_data$sentiment == -1) 


# Calculate the counts of each sentiment level
sentiment_counts <- c(
  sum(twitter_data$sentiment == 2),    
  sum(twitter_data$sentiment == 1),    
  sum(twitter_data$sentiment == 0),    
  sum(twitter_data$sentiment == -1)    
)

# Create a data frame with sentiment labels and counts
sentiment_data <- data.frame(
  sentiment = c("Factual News", "Pro", "Neutral", "Anti"),
  count = sentiment_counts
)

# Calculate the percentage of total tweets for each sentiment level
sentiment_data <- data.frame(
  sentiment = c("2 (News)", "1 (Pro)", "0 (Neutral)", "-1 (Anti)"),
  count = sentiment_counts,
  percentage = round((sentiment_counts / sum(sentiment_counts)) * 100,2)
)

# Plot the sentiment vs tweet counts with percentage labels
ggplot(sentiment_data, aes(x = sentiment, y = count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = paste0(sentiment, "# ", count, " (", percentage, "%)")),
            position = position_stack(vjust = 0.5), color = "black", size = 4) +
  labs(title = "Sentiment vs Tweet Counts", x = "Twitter Sentiment", y = "Number of Tweets") +
  theme_minimal()



### Inference:
##      From the bar chart, we could see the following distribution of sentiment levels among the tweets about climate change:
##
##      ----- 2 (News): The tweets that link to factual news about climate change. Total: 9,276 tweets (21.11%).
##      ----- 1 (Pro): The tweets that support the belief of man-made climate change. Total: 22,962 tweets (52.25%).
##      ----- 0 (Neutral): The tweets that neither support nor refute the belief of man-made climate change. Total: 7,715 tweets (17.56%).
##      ----- -1 (Anti): The tweets that do not believe in man-made climate change. Total: 3,990 tweets (9.08%).




#####################################################################################################
### Step 3: Statistical Analysis: Top N occurrences of tweet words (overall & per sentiment group)
#####################################################################################################


#### Why determine the Top N occurrences of tweet words?
## Determining the top N occurrences of tweet words provides researchers with a valuable tool to summarize, analyze, and understand the content and trends within a large collection of tweets. 
##
## Some common use cases include:
## 1) Data summarization: By identifying the most frequent words or terms in a collection of tweets, researchers can effectively summarize the content and themes of the tweets. 
##    This helps in gaining insights into the popular topics, trends, or discussions happening on the platform.
## 2) Identifying key terms: Analyzing the top N occurrences allows researchers to identify the key terms or hashtags that are being widely used and discussed in the Twitter community. 
##    This information can be used to understand the important concepts and topics related to the research area.
## 3) Comparative analysis: By comparing the top N occurrences across different datasets or time periods, researchers can observe changes in the popularity or prevalence of certain words or topics. 
##    This comparative analysis can provide insights into evolving trends, shifts in public opinion, or emerging discussions in the Twitter sphere.


# To avoid the error: "Error: vector memory exhausted (limit reached?)" while running "word_counts <- rowSums(as.matrix(tdm))", we would need to subset the data as shown below - 
nrow(twitter_data)

length(twitter_data$message[1:100]) 

# Create a vector of stop words, these stop words are determined after executing trial runs of the below
# Add more stop words as needed
filler_words <- c("the", "and", "is", "it", "in", "of", "for", "about", "that", "are", "you", "your", "how", "not", "have", "this",
                  "doesn't","will","with","who", "&amp;","change.", "change,","she's","going","from","but","because", "since", "there",
                  "here", "to", "what", "where", "why", "or", "can", "a","our","more","has", "via", "just", "all","well", "its")  




### Note: To avoid "out-of-memory" errors, the data can be subsetted and processed individually instead of loading the entire dataset at once.

## Subset I: 1:20000

# Remove stop words from the text data
corpus <- Corpus(VectorSource(tolower(twitter_data$message[1:20000])))

# In natural language processing, a corpus refers to a collection of text documents used for analysis. In the above code, the corpus is created 
# using the Corpus function from the tm package. The Corpus function takes a source argument, which in this case is VectorSource(tolower(twitter_data$message[1:20000])). 
# This source argument specifies the text data to be included in the corpus, which is the lowercased text from the first 20,000 messages in the twitter_data dataset.

suppressWarnings(suppressMessages(corpus <- tm_map(corpus, removeWords, filler_words)))

# By using tm_map from the tm package, the removeWords transformation is applied to each document in the corpus object, effectively removing the specified filler words from the text. 
# The result is a modified corpus object with the filler words removed from each document.


# Create a term-document matrix
tdm <- TermDocumentMatrix(corpus)
word_counts <- rowSums(as.matrix(tdm)) # will run for quite some time (~5-10 mins) depending on local machine config

# Sort the word counts in descending order
sorted_counts <- sort(word_counts, decreasing = TRUE)

# Get the top N occurrences
N <- 10
top_N <- head(sorted_counts, N)

# Print the top N occurrences
filtered_top_N_subset1 <- top_N[!(names(top_N) %in% filler_words)]
print(filtered_top_N_subset1)



## Subset II: 20000:43943

# Remove stop words from the text data
corpus <- Corpus(VectorSource(tolower(twitter_data$message[20000:43943])))
suppressWarnings(suppressMessages(corpus <- tm_map(corpus, removeWords, filler_words)))

# Create a term-document matrix
tdm <- TermDocumentMatrix(corpus)
word_counts <- rowSums(as.matrix(tdm)) # will run for quite some time (~5-10 mins) depending on local machine config

# Sort the word counts in descending order
sorted_counts <- sort(word_counts, decreasing = TRUE)

# Get the top N occurrences
N <- 10
top_N <- head(sorted_counts, N)

# Print the top N occurrences
filtered_top_N_subset2 <- top_N[!(names(top_N) %in% filler_words)]
print(filtered_top_N_subset2)


### Merging the two subsets
filtered_top_N_subset1 <- data.frame(filtered_top_N_subset1)  # Convert to data frame

pivoted_data1 <- filtered_top_N_subset1 %>%
  rownames_to_column(var = "word") %>%
  pivot_longer(-word, names_to = "variable", values_to = "count")

print(pivoted_data1)


filtered_top_N_subset2 <- data.frame(filtered_top_N_subset2)  # Convert to data frame

pivoted_data2 <- filtered_top_N_subset2 %>%
  rownames_to_column(var = "word") %>%
  pivot_longer(-word, names_to = "variable", values_to = "count")

print(pivoted_data2)

summary(pivoted_data1)

pivoted_data1 <- pivoted_data1 %>% select(-variable)
pivoted_data2 <- pivoted_data2 %>% select(-variable)

# Merge the datasets based on the "word" column
merged_dataset <- merge(pivoted_data1, pivoted_data2, by = colnames(pivoted_data1), all = TRUE)

# Print the merged dataset
print(merged_dataset)

## Aggregating by word, in order to remove duplicate rows
aggregated_data <- aggregate(count ~ word, merged_dataset, sum)

print(aggregated_data)



### To Plot as a bar chart:

# Calculate the percentage of total tweets for each word
aggregated_data$percentage <- round(aggregated_data$count / sum(aggregated_data$count) * 100, 2)

# Sort the data by percentage in descending order
aggregated_data <- aggregated_data[order(aggregated_data$percentage, decreasing = TRUE), ]


# Create the bar plot
plot1 <- ggplot(aggregated_data, aes(x = word, y = percentage)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = paste0(percentage, "%")), vjust = -0.5, size = 4) +
  labs(title = "Word Counts as Percentage of Total Tweets",
       x = "Word", y = "Percentage") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12))

plot1

################## Generate word cloud

wordcloud(aggregated_data$word, aggregated_data$count, 
          scale=c(7,1), random.order = FALSE, 
          colors = brewer.pal(8, "Dark2")) ## from wordcloud package



### Inference:
##    From the previous outputs and plots, the following statistical inferences can be made:
##    
##    ----- The word "believe" appears 1,653 times, which accounts for 2.69% of the total tweets.
##    ----- The word "change" appears 2,633 times, which accounts for 4.29% of the total tweets.
##    ----- The word "climate" appears 33,458 times, which accounts for 54.54% of the total tweets.
##    ----- The word "global" appears 10,534 times, which accounts for 17.17% of the total tweets.
##    ----- The word "new" appears 820 times, which accounts for 1.34% of the total tweets.
##    ----- The word "people" appears 1,458 times, which accounts for 2.38% of the total tweets.
##    ----- The word "trump" appears 3,517 times, which accounts for 5.73% of the total tweets.
##    ----- The word "warming" appears 7,268 times, which accounts for 11.85% of the total tweets.
##    
##    Further insights from the results are as follows:
##    
##    ----- The word "climate" has the highest count, appearing in 33,458 tweets, which indicates that climate-related discussions are a significant topic of conversation in the analyzed tweets.
##    ----- The words "global" and "warming" also have relatively high counts, with 10,534 and 7,268 occurrences, respectively. This suggests that discussions around global warming and its impact on the environment are prevalent.
##    ----- The word "trump" appears 3,517 times, indicating that there is a notable mention of former President Donald Trump in the context of climate change. This might suggest the impact of Trump's policies or statements on climate-related matters.
##    ----- The word "change" has a substantial count of 2,633, reflecting the emphasis on the concept of change in relation to climate issues.
##    ----- The words "believe," "new," and "people" have relatively lower counts compared to other terms, indicating that they are less frequently mentioned in the analyzed tweets.
##
##    These insights provide a glimpse into the key topics and themes discussed in relation to climate change on Twitter, highlighting the focus on climate, global warming, change, and the involvement of notable figures like Trump.



#################### Top N words vs Sentiments


### What is the underlying objective of classifying the top N words according to their sentiments?
##     The purpose of categorizing the top N words based on their sentiments is to gain insights into the prevailing attitudes and opinions expressed in the text data. 
##     By categorizing the words into different sentiment categories, we can understand the distribution and frequency of positive, negative, or neutral sentiments. 
##     This categorization allows us to analyze sentiment patterns, identify trends, and extract meaningful information about people's opinions, perceptions, and emotions related to the topic of interest. 
##     Such analysis can be valuable for various applications, including sentiment analysis, opinion mining, market research, and understanding public sentiment.

### To categorize the top N words based on their sentiments

# Initialize an empty dataframe
result_df_2 <- data.frame(word = character(), sentiment = numeric(), sum = numeric(), stringsAsFactors = FALSE)

# Filter the data based on the current sentiment
filtered_data <- subset(twitter_data, sentiment == 2)

# Loop through each word in aggregated_data$word
for (word in aggregated_data$word) {
  # Count occurrences of the current word
  count_word <- str_count(filtered_data$message, paste0("\\b", word, "\\b"))
  
  # Sum the occurrences
  sum_word <- sum(count_word)
  
  # Add the word, sentiment, count, and sum to the result dataframe
  result_df_2 <- rbind(result_df_2, data.frame(word = word, sentiment = 2, sum = sum_word))
}


# Print the result dataframe
print(result_df_2)


# Initialize an empty dataframe
result_df_1 <- data.frame(word = character(), sentiment = numeric(), sum = numeric(), stringsAsFactors = FALSE)

# Filter the data based on the current sentiment
filtered_data <- subset(twitter_data, sentiment == 1)

# Loop through each word in aggregated_data$word
for (word in aggregated_data$word) {
  # Count occurrences of the current word
  count_word <- str_count(filtered_data$message, paste0("\\b", word, "\\b"))
  
  # Sum the occurrences
  sum_word <- sum(count_word)
  
  # Add the word, sentiment, count, and sum to the result dataframe
  result_df_1 <- rbind(result_df_1, data.frame(word = word, sentiment = 1, sum = sum_word))
}


# Print the result dataframe
print(result_df_1)


# Initialize an empty dataframe
result_df_0 <- data.frame(word = character(), sentiment = numeric(), sum = numeric(), stringsAsFactors = FALSE)

# Filter the data based on the current sentiment
filtered_data <- subset(twitter_data, sentiment == 0)

# Loop through each word in aggregated_data$word
for (word in aggregated_data$word) {
  # Count occurrences of the current word
  count_word <- str_count(filtered_data$message, paste0("\\b", word, "\\b"))
  
  # Sum the occurrences
  sum_word <- sum(count_word)
  
  # Add the word, sentiment, count, and sum to the result dataframe
  result_df_0 <- rbind(result_df_0, data.frame(word = word, sentiment = 0, sum = sum_word))
}


# Print the result dataframe
print(result_df_0)



# Initialize an empty dataframe
result_df_minus1 <- data.frame(word = character(), sentiment = numeric(), sum = numeric(), stringsAsFactors = FALSE)

# Filter the data based on the current sentiment
filtered_data <- subset(twitter_data, sentiment == -1)

# Loop through each word in aggregated_data$word
for (word in aggregated_data$word) {
  # Count occurrences of the current word
  count_word <- str_count(filtered_data$message, paste0("\\b", word, "\\b"))
  
  # Sum the occurrences
  sum_word <- sum(count_word)
  
  # Add the word, sentiment, count, and sum to the result dataframe
  result_df_minus1 <- rbind(result_df_minus1, data.frame(word = word, sentiment = -1, sum = sum_word))
}


# Print the result dataframe
print(result_df_minus1)



result_df <- dplyr::bind_rows(result_df_2, result_df_1, result_df_0, result_df_minus1)

print(result_df)




# Create a new plot window
par(mfrow = c(2, 2))

# Sort the data by sum in descending order
result_df_sorted <- result_df[order(result_df$sum, decreasing = TRUE), ]

# Plot the first axis
barplot(result_df_sorted$sum[result_df_sorted$sentiment == 2], 
        names.arg = result_df_sorted$word[result_df_sorted$sentiment == 2],
        main = "Sentiment 2 (News): the tweet links to factual news about climate change", xlab = "Word", ylab = "Sum", col = "purple",
        ylim = c(0, max(result_df_sorted$sum)), las = 2)
text(x = 1:length(result_df_sorted$word[result_df_sorted$sentiment == 2]),
     y = result_df_sorted$sum[result_df_sorted$sentiment == 2],
     labels = result_df_sorted$sum[result_df_sorted$sentiment == 2], pos = 3)

# Plot the second axis
barplot(result_df_sorted$sum[result_df_sorted$sentiment == 1], 
        names.arg = result_df_sorted$word[result_df_sorted$sentiment == 1],
        main = "Sentiment 1(Pro): the tweet supports the belief of man-made climate change", xlab = "Word", ylab = "Sum", col = "green",
        ylim = c(0, max(result_df_sorted$sum)), las = 2)
text(x = 1:length(result_df_sorted$word[result_df_sorted$sentiment == 1]),
     y = result_df_sorted$sum[result_df_sorted$sentiment == 1],
     labels = result_df_sorted$sum[result_df_sorted$sentiment == 1], pos = 3)

# Plot the third axis
barplot(result_df_sorted$sum[result_df_sorted$sentiment == 0], 
        names.arg = result_df_sorted$word[result_df_sorted$sentiment == 0],
        main = "Sentiment 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change", xlab = "Word", ylab = "Sum", col = "blue",
        ylim = c(0, max(result_df_sorted$sum)), las = 2)
text(x = 1:length(result_df_sorted$word[result_df_sorted$sentiment == 0]),
     y = result_df_sorted$sum[result_df_sorted$sentiment == 0],
     labels = result_df_sorted$sum[result_df_sorted$sentiment == 0], pos = 3)

# Plot the fourth axis
barplot(result_df_sorted$sum[result_df_sorted$sentiment == -1], 
        names.arg = result_df_sorted$word[result_df_sorted$sentiment == -1],
        main = "Sentiment -1(Anti): the tweet does not believe in man-made climate change", xlab = "Word", ylab = "Sum", col = "red",
        ylim = c(0, max(result_df_sorted$sum)), las = 2)
text(x = 1:length(result_df_sorted$word[result_df_sorted$sentiment == -1]),
     y = result_df_sorted$sum[result_df_sorted$sentiment == -1],
     labels = result_df_sorted$sum[result_df_sorted$sentiment == -1], pos = 3)


### Inference:
##     The top 3 words from each sentiment category:
##
##     1) News Sentiment: [2 - News: the tweet links to factual news about climate change]
##        - "climate" - 7,792 tweets
##        - "change" - 7,756 tweets
##        - "global" - 1,284 tweets
##
##       The presence of words like "climate," "change," and "global" indicates that news tweets about climate change often focus on these topics. 
##       The relatively high count of tweets containing these words suggests a significant amount of factual news content related to climate change.
##
##     2) Positive Sentiment: [1 - Pro: the tweet supports the belief of man-made climate change]
##        - "climate" - 18,767 tweets
##        - "change" - 18,712 tweets
##        - "global" - 4,011 tweets
##
##       The repeated occurrence of words like "climate," "change," and "global" in positive sentiment tweets signifies strong support for the belief in man-made climate change.
##       The higher tweet counts for these words indicate a positive attitude towards addressing climate change and promoting awareness.
##  
##     3) Neutral Sentiment: [0 - Neutral: the tweet neither supports nor refutes the belief of man-made climate change]
##        - "change" - 4,277 tweets
##        - "climate" - 4,255 tweets
##        - "global" - 3,032 tweets
##       
##       The occurrence of words like "climate," "change," and "global" in neutral sentiment tweets suggests that these topics are commonly discussed without taking a specific stance.
##       The similar tweet counts for these words indicate a balanced representation of neutral opinions on climate change.
##      
##     4) Negative Sentiment: [-1 - Anti: the tweet does not believe in man-made climate change]
##        - "climate" - 2,121 tweets
##        - "change" - 2,083 tweets
##        - "global" - 1,733 tweets
##       
##       The presence of words like "climate," "change," and "global" in negative sentiment tweets suggests skepticism or disbelief in man-made climate change.
##       The lower tweet counts for these words indicate a relatively smaller proportion of tweets expressing negative sentiments towards climate change.
##    




###########################################################################################
### Step 4: Sentiment Analysis: How well does the sentiment scores reflect user sentiment?
###########################################################################################

#### What is the purpose of Sentiment analysis?
##      Sentiment analysis is performed to understand and quantify the sentiment or emotional tone expressed in a piece of text, such as tweets, reviews, or comments. 
##      It involves analyzing the text to determine whether it expresses a positive, negative, or neutral sentiment. 
##      The goal is to extract insights about people's opinions, attitudes, or emotions towards a particular topic or entity.

#### What are Sentiment scores?
##      Sentiment scores, also known as polarity scores, are numerical values assigned to each piece of text to represent the sentiment expressed. These scores indicate the degree of positivity or negativity in the text. 
##      Typically, sentiment scores range from -1 to 1, where -1 represents strong negative sentiment, 1 represents strong positive sentiment, and 0 represents neutral sentiment.
##
##      For example, let's consider a tweet: "I absolutely loved the new movie! The acting was brilliant, and the storyline kept me engaged throughout." 
##      In this case, a sentiment analysis algorithm would assign a positive sentiment score close to 1, indicating that the tweet expresses a highly positive sentiment. 
##      The algorithm would analyze the words "loved," "brilliant," and "engaged" to determine the positive sentiment.



# Create a function to calculate sentiment scores
calculate_sentiment_score <- function(text) {
  sentiment <- sentiment(text, polarity_dt = lexicon::hash_sentiment_jockers_rinker)
  mean_sentiment <- mean(sentiment$sentiment)
  return(mean_sentiment)
} # from sentimentr package

# Apply the sentiment score calculation to the 'message' column in the data frame
twitter_data$sentiment_score <- sapply(twitter_data$message, calculate_sentiment_score) # will run for quite some time (~15-25 mins) depending on local machine config

# View the updated data frame
head(twitter_data,10)

nrow(twitter_data)



head(twitter_data[order(twitter_data$sentiment_score, decreasing = TRUE), ],10)

head(twitter_data[order(twitter_data$sentiment_score, decreasing = FALSE), ],10)



#### Inference:
## The output displays the top 10 rows of the twitter_data dataframe, sorted by the sentiment_score column in both ascending and descending order. Each row contains the sentiment, message, tweetid, and sentiment_score values.
## 
## These results provide insights into the sentiment of the tweets in the dataset, with higher sentiment scores indicating more positive sentiment and lower scores indicating more negative sentiment. 
## It allows for further analysis and understanding of the sentiment distribution in the dataset.
##
## Analyzing the relationship between the sentiment_score and message columns in the twitter_data dataframe reveals additional insights:
##
##   ---- Tweets with Positive Sentiment scores: In general, Tweets with higher sentiment scores often contain positive language, expressing support, enthusiasm, or agreement with regards to climate change. 
##                            These tweets may contain words and phrases indicating optimism, proactive actions, and belief in the urgency of addressing climate change. 
##                            However, it's important to note that there are tweet messages with a positive sentiment score that express disbelief or skepticism towards the notion of climate change. 
##                            This can be attributed to the nature of sentiment analysis using natural language processing (NLP), where the sentiment score is determined based on the overall sentiment conveyed by the words and phrases used in the message. 
##                            In such cases, although the sentiment score may indicate positivity, the actual sentiment expressed in the message may differ, highlighting the complexities and nuances of sentiment analysis in capturing the true sentiment behind a text.
##
##   ---- Tweets with Negative Sentiment scores: Tweets with lower sentiment scores tend to have negative language, reflecting skepticism, denial, or criticism of climate change. 
##                           These tweets may contain words and phrases indicating disbelief, questioning the validity of climate change, or expressing negative emotions towards related topics.
##                           Also, as mentioned before, there are tweet messages with a negative sentiment score that may convey support or agreement with the concept of climate change. 
##                           This is because sentiment analysis using natural language processing (NLP) assigns sentiment scores based on the language and context used in the text. 
##                           In some cases, negative sentiment scores may be given to tweets that discuss the negative impacts or consequences of climate change, even though the overall sentiment expressed in the 
##                           message aligns with the belief in climate change. This highlights the challenges of sentiment analysis and the importance of considering the context and nuances of the text when interpreting sentiment scores.
##


#### Scatter plot for showing the sentiment scores distribution
ggplot(twitter_data, aes(x = 1:length(sentiment_score), y = sentiment_score, color = cut(sentiment_score, breaks = c(-Inf, -0.1, 0.1, Inf), labels = c("Negative", "Neutral", "Positive")))) +
  geom_point() +
  labs(title = "Sentiment Scores distribution", x = "Index", y = "Sentiment Score") +
  scale_color_manual(values = c("red", "blue", "green"))


#### Box plot for outlier detection
ggplot(twitter_data, aes(x = cut(sentiment_score, breaks = c(-Inf, -0.1, 0.1, Inf), labels = c("Negative", "Neutral", "Positive")), y = sentiment_score, 
                         fill = cut(sentiment_score, breaks = c(-Inf, -0.1, 0.1, Inf), labels = c("Negative", "Neutral", "Positive")))) +
  geom_boxplot() +
  geom_text(data = filter(twitter_data, sentiment_score == max(sentiment_score)), aes(label = sentiment_score), hjust = 0, vjust = 1.5, color = "black", size = 3) +
  geom_text(data = filter(twitter_data, sentiment_score == min(sentiment_score)), aes(label = sentiment_score), hjust = 0, vjust = 1.5, color = "black", size = 3) +
  labs(title = "Sentiment Scores Boxplot", x = "Sentiment Category", y = "Sentiment Score") +
  scale_fill_manual(values = c("red", "blue", "green"))

#### Inferences:
## The scatter plot visualizes the distribution of sentiment scores in the Twitter data. The x-axis represents the index of the tweets, and the y-axis represents the sentiment scores. 
## The sentiment scores are color-coded into three categories: "Negative," "Neutral," and "Positive." The plot shows the distribution of sentiment scores across the tweets. 
##
## The box plot is used for outlier detection in the sentiment scores. The x-axis represents the sentiment categories ("Negative," "Neutral," and "Positive"), and the y-axis represents the sentiment scores. 
## The fill color of the boxes corresponds to the sentiment categories. The plot also includes text labels indicating the maximum sentiment score for both positive and negative outliers.
## 
## The analysis from the scatter plot indicates that the negative sentiment scores and positive sentiment scores are evenly distributed across the tweets. This means that there is a relatively equal volume of tweets with negative 
## sentiment and tweets with positive sentiment. The distribution suggests that there is a balance between expressions of negative and positive sentiments regarding the topic being analyzed (e.g., climate change).
##
## The analysis from the boxplot reveals the presence of outliers in both negative and positive sentiment scores. The maximum positive outlier is at 1.29, while the maximum negative outlier is at -1.83. 
## These outliers indicate extreme sentiment scores that deviate significantly from the majority of the sentiment scores in their respective categories.




###############################################################################################################################
### ************************************************ Step 5: Conclusion *******************************************************
###############################################################################################################################

############### Statistical Analysis: Top N occurrences of tweet words (overall & per sentiment group)
### 
### The analysis of Twitter tweets on climate change revealed several key insights. From the word frequency analysis, it was found that "climate" is the most commonly mentioned word, 
### appearing in 33,458 tweets, followed by "global" with 10,534 occurrences. This indicates that climate-related discussions are highly prevalent on Twitter. The word "trump" was 
### mentioned in 3,517 tweets, suggesting his impact on climate-related matters. The word "change" had 2,633 occurrences, emphasizing the focus on the concept of change in the context of climate issues.
###
### Further exploration based on sentiment revealed the top 3 words associated with each sentiment category. For news sentiment, "climate," "change," and "global" were frequently used, indicating a focus 
### on factual news about climate change. Positive sentiment tweets had repeated occurrences of "climate," "change," and "global," showing strong support for man-made climate change belief. Neutral sentiment 
### tweets mentioned "climate," "change," and "global" without taking a specific stance, reflecting balanced representation. Negative sentiment tweets also featured "climate," "change," and "global," indicating 
### skepticism or disbelief in man-made climate change.
###
### Overall, this analysis provides valuable insights into the key topics and themes of climate change discussions on Twitter, highlighting the prevalence of climate-related content and the varying sentiments expressed by users.


############### Sentiment Analysis: How well does the sentiment scores reflect user sentiment?
### 
### The displayed output provides insightful observations on sentiment analysis results and the relationship between sentiment scores and tweet messages. The twitter_data dataframe's top 10 rows are sorted by sentiment_score, 
### revealing tweets with higher sentiment scores expressing positive sentiments, while lower scores convey negative sentiments. However, it's important to note that some positive sentiment tweets may express disbelief or skepticism towards 
### climate change, and vice versa for negative sentiment tweets. This highlights the complexities of sentiment analysis using NLP. The scatter plot illustrates the distribution of sentiment scores, showing an even volume of negative and positive 
### sentiment tweets. The box plot identifies outliers with extreme sentiment scores i.e., maximum positive outlier is at 1.29, while the maximum negative outlier is at -1.83, indicating significant deviations from the majority. 
###
### Overall, the analysis provides valuable insights into sentiment distribution and the challenges of sentiment analysis in capturing the true sentiment behind text data.


