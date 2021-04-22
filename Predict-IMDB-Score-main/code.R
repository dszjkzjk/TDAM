
##### IMPORTING IMPORTANT LIBRARIES
#library(tidyverse)
library(car)
library(splines)
library(broom)
library(boot)
library(caTools)
library(dplyr)

###### IMPORTING DATASET #########

clean_movie = read.csv("/Users/zhangjunkang/Downloads/MMA 2019/Courses/MGSC661/Midterm Project/clean_movie_v1.csv")
attach(clean_movie)

##### Include code to change the countries column

############# REMOVE EXTRA COLUMNS #######

#delete all the columns we don't need (column color in yellow)
columns_not_required <-c("X","movie_id",
                         "title",
                        "release_day",
                         "release_year",
                         "budget","language",
                         "content_rating",
                         "aspect_ratio",
                         "actor_1_name",
                         "actor_1_facebook_likes","actor_1_known_for",
                         "actor_2_name",
                         "actor_2_facebook_likes","actor_2_known_for",
                        "actor_3_name",
                         "actor_3_facebook_likes","actor_3_known_for",
                         "actor_3_star_meter","movie_imdb_link",
                         "color",
                        "action","adventure",
                        "scifi","thriller",
                        "musical","romance",
                        "western","sport",
                        "horror","drama",
                        "war","animation",
                        "crime",
                         "plot_keywords",
                         "ratio_movie_cast_likes",
                         "number_of_votes","cinematographer",
                         "production_company","plot_summary",
                         "love","friend",
                         "murder","death",
                         "police","high.school",
                         "new.york.city","alien",
                         "fbi","drugs")

clean_movie <- clean_movie %>% select(-columns_not_required)

################### Create dummy variables for genre to include all genres


y3<-clean_movie[c("imdb_score","genres")]%>% transform(genres = strsplit(as.character(genres), "|",fixed=TRUE)) %>% unnest(genres)
temp<-y3 %>% group_by(genres) %>% summarise(mean=mean(imdb_score),count=n()) %>% arrange(desc(mean))

genres_to_create <-temp$genres[!(temp$genres %in%c("Film-Noir","Short") ) ] 

str_detect(clean_movie$genres,i)
for (i in genres_to_create){
  temp1<-str_detect(as.character(clean_movie$genres),i)
  clean_movie[temp1,i]<-1
  clean_movie[!temp1,i]<-0
}

summary(clean_movie)

############### Run R2 test on the following columns with respect to imdb_score

#1. Correlation matrix for continous variables
numeric_cols<-c()
for( i in names(clean_movie)){
  if(is.numeric(clean_movie[[i]])){
    numeric_cols <-c(numeric_cols,i)
  }
}


#the following code tries to find which simple linear regression model has the highest r-squared
name_list = colnames(clean_movie)

rSquared = NULL
for(i in names(clean_movie) ){ 
  if (is.numeric(clean_movie[[i]])){ 
    reg = lm(imdb_score~clean_movie[[i]], data = clean_movie) 
    Sreg = summary(reg) 
    rSquared = c(rSquared, Sreg$r.squared, i) 
  }
}

rSquared = matrix(rSquared, byrow = TRUE, ncol = 2)
write.csv(rSquared,"rsquared.csv")

# colrel matrix
cor_matrix1 = cor(clean_movie[numeric_cols],use = "pairwise.complete.obs")
write.csv(cor_matrix1, "cor_matrix_temp.csv")

# to check variance inflation
as.data.frame(vif(lm(formula =imdb_score~. ,data = clean_movie[numeric_cols])),col.names = c("name","value"))

############################### REMOVE VARIABLES ##############

############## REMOVE CATEGORICAL VARIABLES
 
categorical_variables<-c()
for( i in names(clean_movie)){
  if(!is.numeric(clean_movie[[i]])){
    categorical_variables <-c(categorical_variables,i)
  }
}

clean_movie_modified <-clean_movie %>% select(-categorical_variables) 


##########################  To remove variables that are correlated
###############  FIRST ROUND REMOVAL OF VARIABLES
first_round_remove <- c("cast_total_facebook_likes","number_news_articles","user_reviews_number","movie_facebook_likes")
  # If we see cor_matrix_temp.csv, we can see that these variabes have high collinearity with other variables that have high r2. hence it is important to remove them

clean_movie %>% select(-first_round_remove) 
vif(lm(formula =imdb_score~. ,data = clean_movie[numeric_cols] %>% select(-first_round_remove) ))

clean_movie_modified <-clean_movie_modified %>% select(-first_round_remove) 

###############  SECOND ROUND REMOVAL OF VARIABLES
numeric_cols1<-c()
for( i in names(clean_movie_modified)){
  if(is.numeric(clean_movie_modified[[i]])){
    numeric_cols1 <-c(numeric_cols1,i)
  }
}

cor_matrix1 = cor(clean_movie_modified[numeric_cols1],use = "pairwise.complete.obs")
write.csv(cor_matrix1, "cor_matrix_modified.csv")

# As seen in cor_matrix_modified.csv, we can see that movie_budget and adventure ae correlated..so we remove adventure..critic_review_number and movie_budget are clashing..so lets remove that as well

second_round_remove <- c("Adventure","movie_budget")
clean_movie_modified <-clean_movie_modified %>% select(-second_round_remove) 

###############  THIRD ROUND REMOVAL OF VARIABLES
vif(lm(formula =imdb_score~. ,data = clean_movie_modified[numeric_cols1]  ))


    # Remove them if necessary later

########## OUTLIER REMOVAL-- PERFORM IF NECESSARY

plot(clean_movie$number_of_faces_in_movie_poster,clean_movie$imdb_score)
augment(lm(formula = imdb_score~number_of_faces_in_movie_poster,data = clean_movie))%>% arrange(desc(.cooksd)) %>%head()
clean_movie[1825,"number_of_faces_in_movie_poster"]






############################## TEST LINEARITY of PREDICTORS ###############


# Run simple linear regression on all predictors to see which is linear and which is not

name_list=names(clean_movie_modified)
tukey_p = NULL
for (i in 1:length(names(clean_movie_modified))) {
  if (is.numeric(clean_movie_modified[1,i])){ 
    reg_tukey = lm(imdb_score~clean_movie_modified[[i]], data = clean_movie_modified) 
    plot = residualPlots(reg_tukey, xlab = name_list[i])
    tukey_p = c(tukey_p, plot[1,2], name_list[i])
  }
}
tukey_p = matrix(tukey_p, byrow = TRUE, ncol = 2)

#see which variable is not linear
non_linear_cols <- c()
for (i in 1:length(tukey_p[,1])) {
  if (as.numeric(tukey_p[i,1]) < 0.05) {
    non_linear_cols <-c(non_linear_cols,tukey_p[i,2])
    print(tukey_p[i,2])
  }
}

################### REMOVE ALL NA's IN THE DATASET
clean_movie_modified<-clean_movie_modified %>% na.exclude()
  # Find a better way to treat NA's later

######## FIND THE BEST POLYNOMIAL FOR NON-LINEAR COLUMNS

######### Automated code for all non_linear_cols-- producing warnings
for(i in non_linear_cols){
  cv.error = rep(0,9)
  for (j in 2:10){
    glmfit=glm(clean_movie_modified$imdb_score~poly(clean_movie_modified[[i]],j))
    cv.error[j-1]=cv.glm(clean_movie_modified,glmfit,K=10)$delta[1]
  }
  print(paste0("The best degree for ",i," is ",which.min(cv.error)+1))
}

########## manual code for each of the non-linear cols
cv.error = rep(0,9)
for (i in 2:10){
  glmfit=glm(imdb_score~poly(duration_mins,i),data=clean_movie_modified)
  cv.error[i-1]=cv.glm(clean_movie_modified,glmfit,K=10)$delta[1]
}
which.min(cv.error)+1
  # Says that 8th degree is the best for duration_mins

cv.error = rep(0,9)
for (i in 2:10){
  glmfit=glm(imdb_score~poly(critic_reviews_number,i),data=clean_movie_modified)
  cv.error[i-1]=cv.glm(clean_movie_modified,glmfit,K=10)$delta[1]
}
which.min(cv.error)+1
# Says that 7th degree is the best for critic_reviews_number

cv.error = rep(0,9)
for (i in 2:10){
  glmfit=glm(imdb_score~poly(user_votes_number,i),data=clean_movie_modified)
  cv.error[i-1]=cv.glm(clean_movie_modified,glmfit,K=10)$delta[1]
}
which.min(cv.error)+1
# Says that 7th degree is the best for user_votes_number


cv.error = rep(0,9)
for (i in 2:10){
  glmfit=glm(imdb_score~poly(sum_total_likes,i),data=clean_movie_modified)
  cv.error[i-1]=cv.glm(clean_movie_modified,glmfit,K=10)$delta[1]
}
which.min(cv.error)+1
# Says that 3th degree is the best for sum_total_likes

cv.error = rep(0,9)
for (i in 2:10){
  glmfit=glm(imdb_score~poly(movie_meter_IMDB_pro,i),data=clean_movie_modified)
  cv.error[i-1]=cv.glm(clean_movie_modified,glmfit,K=10)$delta[1]
}
which.min(cv.error)+1
# Says that 3rd degree is the best for movie_meter_IMDB_pro

######################### PERFORM ANOVA TEST TO PICK THE OPTIMIZED DEGREE FOR EACH OF THE NON-LINEAR PREDICTORS

### Duration mins
summary(clean_movie_modified)
reg2=lm(imdb_score~poly(duration_mins,2), data=clean_movie_modified)
reg3=lm(imdb_score~poly(duration_mins,3), data=clean_movie_modified)
reg4=lm(imdb_score~poly(duration_mins,4), data=clean_movie_modified)
reg5=lm(imdb_score~poly(duration_mins,5), data=clean_movie_modified)
reg6=lm(imdb_score~poly(duration_mins,6), data=clean_movie_modified)
anova(reg2,reg3,reg4,reg5,reg6)
#pick degree 4 based on p-value and rss

### critical reviews number
reg2_c=lm(imdb_score~poly(critic_reviews_number,2), data=clean_movie_modified)
reg3_c=lm(imdb_score~poly(critic_reviews_number,3), data=clean_movie_modified)
reg4_c=lm(imdb_score~poly(critic_reviews_number,4), data=clean_movie_modified)
reg5_c=lm(imdb_score~poly(critic_reviews_number,5), data=clean_movie_modified)
reg6_c=lm(imdb_score~poly(critic_reviews_number,6), data=clean_movie_modified)
anova(reg2_c,reg3_c,reg4_c,reg5_c,reg6_c)
#pick degree 3 based on p-value and rss

###user votes number
reg1_u=lm(imdb_score~user_votes_number,data=clean_movie_modified)
reg2_u=lm(imdb_score~poly(user_votes_number,2), data=clean_movie_modified)
reg3_u=lm(imdb_score~poly(user_votes_number,3), data=clean_movie_modified)
reg4_u=lm(imdb_score~poly(user_votes_number,4), data=clean_movie_modified)
reg5_u=lm(imdb_score~poly(user_votes_number,5), data=clean_movie_modified)
reg6_u=lm(imdb_score~poly(user_votes_number,6), data=clean_movie_modified)
anova(reg2_u,reg3_u,reg4_u,reg5_u,reg6_u)
summary(reg1_u)
#pick degree 3 based on p-value and rss

###sum total likes
reg2_s=lm(imdb_score~poly(sum_total_likes,2), data=clean_movie_modified)
reg3_s=lm(imdb_score~poly(sum_total_likes,3), data=clean_movie_modified)
reg4_s=lm(imdb_score~poly(sum_total_likes,4), data=clean_movie_modified)
reg5_s=lm(imdb_score~poly(sum_total_likes,5), data=clean_movie_modified)
reg6_s=lm(imdb_score~poly(sum_total_likes,6), data=clean_movie_modified)
anova(reg2_s,reg3_s,reg4_s,reg5_s,reg6_s)
#pick degree 3 based on p-value and rss

###movies_meter_imdb_pro
reg1_m=lm(imdb_score~movie_meter_IMDB_pro, data=clean_movie_modified)
reg2_m=lm(imdb_score~poly(movie_meter_IMDB_pro,2), data=clean_movie_modified)
reg3_m=lm(imdb_score~poly(movie_meter_IMDB_pro,3), data=clean_movie_modified)
reg4_m=lm(imdb_score~poly(movie_meter_IMDB_pro,4), data=clean_movie_modified)
reg5_m=lm(imdb_score~poly(movie_meter_IMDB_pro,5), data=clean_movie_modified)
reg6_m=lm(imdb_score~poly(movie_meter_IMDB_pro,6), data=clean_movie_modified)
reg7_m=lm(imdb_score~poly(movie_meter_IMDB_pro,7), data=clean_movie_modified)
reg8_m=lm(imdb_score~poly(movie_meter_IMDB_pro,8), data=clean_movie_modified)
anova(reg1_m,reg2_m,reg3_m,reg4_m,reg5_m,reg6_m,reg7_m,reg8_m)
summary(reg1_m)
#pick degree 3 based on p-value and rss

#################### BUILD A MODEL WITH ALL THE PREDICTORS
names(clean_movie_modified)
names(clean_movie_modified)[25] <- "Scifi"
non_linear_cols
linear_cols<-names(clean_movie_modified)[!names(clean_movie_modified) %in% non_linear_cols]

sample=sample.split(Y =clean_movie_modified$imdb_score,SplitRatio = 1/2)
train=clean_movie_modified[sample,] ### FIgure out why this step introduces additional NA's
test=clean_movie_modified[sample==FALSE,]### FIgure out why this step introduces additional NA's
train<-train %>% na.exclude()
test<-test %>% na.exclude()

final_model<-lm(data = train,formula = imdb_score~ poly(duration_mins,2)+poly(critic_reviews_number,2)+poly(user_votes_number,2)+ poly(sum_total_likes,2)+movie_meter_IMDB_pro+director_facebook_likes + actor_1_star_meter + actor_2_star_meter + number_of_faces_in_movie_poster + Documentary +History + Biography + War + Western + Drama + Animation + Sport +Crime +Mystery +   Musical + Romance + Thriller + Music + Fantasy + Action + Family + Comedy +  Horror)

#summary(final_model)

# Predict on the test set

test$pred=predict(final_model,test)
test$res = (test$imdb_score-test$pred)
test$square = test$res^2
mean(test$square)
summary(test$pred)
test[which.min(test$pred),]


################# TRY OUT BY REMOVING SOME PREDICTORS --- TRIAL 1

# Takeaway 1 : If we remove critic_reviews_number, MSE increases.. but we definitely need to have user_votes_number, MSE:0.55-0.66 
sample=sample.split(Y =clean_movie_modified$imdb_score,SplitRatio = 5/6)
train=clean_movie_modified[sample,] ### FIgure out why this step introduces additional NA's
test=clean_movie_modified[sample==FALSE,]### FIgure out why this step introduces additional NA's
train<-train %>% na.exclude()
test<-test %>% na.exclude()

final_model<-lm(data = train,formula = imdb_score~ poly(duration_mins,2)+poly(user_votes_number,2)+ poly(sum_total_likes,2)+movie_meter_IMDB_pro+director_facebook_likes + actor_1_star_meter + actor_2_star_meter + number_of_faces_in_movie_poster + Documentary +History + Biography + War + Western + Drama + Animation + Sport +Crime +Mystery +   Musical + Romance + Thriller + Music + Fantasy + Action + Family + Comedy +  Horror + Scifi)

test$pred=predict(final_model,test)
test$res = (test$imdb_score-test$pred)
test$square = test$res^2
mean(test$square)
summary(test$pred)
test[which.min(test$pred),]

################# TRY OUT BY REMOVING SOME PREDICTORS --- TRIAL 2

# Here we try removing useless predictors which have less importance. 
#Range of MSE : 0.57-0.67, there is a 0.01 increase in MSE from Trial 1

clean_movie_modified_1<-clean_movie_modified %>% select(-c("Animation",Sport,War,Western,History,Musical,Mystery))
sample=sample.split(Y =clean_movie_modified_1$imdb_score,SplitRatio = 5/6)
train=clean_movie_modified_1[sample,] ### FIgure out why this step introduces additional NA's
test=clean_movie_modified_1[sample==FALSE,]### FIgure out why this step introduces additional NA's
train<-train %>% na.exclude()
test<-test %>% na.exclude()

final_model_1<-lm(data = train,formula = imdb_score~ poly(duration_mins,2)+poly(user_votes_number,2)+ poly(sum_total_likes,2)+movie_meter_IMDB_pro+director_facebook_likes + actor_1_star_meter + actor_2_star_meter + number_of_faces_in_movie_poster + Documentary + Biography + Drama  +Crime  + Romance + Thriller + Music + Fantasy + Action + Family + Comedy +  Horror + Scifi)

test$pred=predict(final_model_1,test)
test$res = (test$imdb_score-test$pred)
test$square = test$res^2
mean(test$square)
summary(test$pred)
test[which.min(test$pred),]

vif(lm(formula =imdb_score~. ,data = clean_movie_modified_1))

cor_matrix1 = cor(clean_movie_modified_1,use = "pairwise.complete.obs")
write.csv(cor_matrix1, "cor_matrix_1.csv")

######################## TRY INTERACTION TERMS

######################## 


######################## RANDOM FOREST REGRESSOR --Simply run and see what happens

library(randomForest)

sample=sample.split(Y =clean_movie_modified$imdb_score,SplitRatio = 5/6)
train=clean_movie_modified[sample,] ### FIgure out why this step introduces additional NA's
test=clean_movie_modified[sample==FALSE,]### FIgure out why this step introduces additional NA's
train<-train %>% na.exclude()
test<-test %>% na.exclude()

regressor = randomForest(x = train%>%select(-c(imdb_score)),
                         y = train$imdb_score,
                         ntree = 40)

# Predicting a new result with Random Forest Regression
y_pred = predict(regressor, test)
test$pred_forest=predict(regressor,test)
test$res_forest = (test$imdb_score-test$pred_forest)
test$square_forest = test$res_forest^2
mean(test$square_forest)

##### Importance of Predictors given by our beautiful Random Forest
importance(regressor) %>% 
  data.frame() %>% 
  mutate(feature = row.names(.))  %>% arrange(desc(IncNodePurity))

names(clean_movie)


############# RANDOM FOREST REGRESSOR ON ENTIRE DATASET
clean_movie<-clean_movie %>% na.exclude()
clean_movie_categorical_removed<-clean_movie %>% select(-c("distributor","director","genres")) 

sample1=sample.split(Y =clean_movie_categorical_removed$imdb_score,SplitRatio = 2/3)
train1=clean_movie_categorical_removed[sample==TRUE,]
test1=clean_movie_categorical_removed[sample==FALSE,]
train1<-train1 %>% na.exclude()
test1<-test1 %>% na.exclude()

regressor1 = randomForest(x = train1%>%select(-imdb_score),
                         y = train1$imdb_score,
                         ntree = 500)

# Predicting a new result with Random Forest Regression

test1$pred_forest=predict(regressor1,test1)
test1$res_forest = (test1$imdb_score-test1$pred_forest)
test1$square_forest = test1$res_forest^2
mean(test1$square_forest)

## Importance of predictors
importance(regressor1) %>% 
  data.frame() %>% 
  mutate(feature = row.names(.))  %>% arrange(desc(IncNodePurity))



############################# PREDICTING PREDICTORS ##############

need_to_predict <- c("user_votes_number","critic_reviews_number")
