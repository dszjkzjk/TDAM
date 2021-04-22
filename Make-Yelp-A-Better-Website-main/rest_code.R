################## start here
library(dplyr)
business = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/new_business.csv')
business_sample = business[1:1000,]
business_sample = business_sample%>%mutate(is_hairsalon = grepl("Hair Salons",business_sample$categories))
business_hairsalon = business_sample %>% filter(business_sample$is_hairsalon == TRUE)


business = business%>%mutate(is_restaurant = grepl(paste(c("Hotel",""),sep="|"),business$categories))
business_rest = business %>% filter(business$is_restaurant == TRUE)

business = business%>%mutate(is_salon = grepl(paste(c("Salon","Spas"),sep="|"),business$categories))
business_salon = business %>% filter(business$is_salon == TRUE)

business = business%>%mutate(is_hotel = grepl(paste(c("Hotels","Inn"),sep="|"),business$categories))
business_hotel = business %>% filter(business$is_hotel == TRUE)

write.csv(business_rest,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest.csv')
write.csv(business_hotel,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_hotel.csv')

############## to python to merge with review

########### to R
########### pick text key words， from lindsay
rest_keywords = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/part3/rest_keywords.csv')
a = rest_keywords %>% filter(rest_keywords$freq >=100)
avector <- rest_keywords[,2]
class(avector)
as.vector(unlist(a$word))

########################### check keywords in each review and create dummies for each keyword
########## grepl review text
#business_rest_hrhs_review = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest_hrhs_review.csv')

business_rest1000_review100_review = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest1000_review100_review.csv')

high_star_kw= c('great','delicious','best','chicken','amazing','staff','server','fresh','friendly','salad','people', 'meat','favourite','sweet','atmosphere','awesome','hot','tasty','excellent','new','pork','perfect','happy','quality','special','wife','husband','portions','selection','attentive','green','seating','clean','wonderful','flavorful','decent','style','group','garlic','quickly','extremely','garlic','ramen','tender','decent','flavorful','wonderful','crispy','crispy','yummy','fan','free','fantastic','fun','cool','quick','perfectly','attentive','family','waiter','enjoyed','location','spicy','price','quite','home','busy','waitress','top','thai')

high_star_kw = unique(high_star_kw)

#business_rest_hrhs_review = business_rest_hrhs_review %>%mutate(is_great = grepl(high_star_kw[1],business_rest_hrhs_review$text))

#########################
for(i in 1:64){ 
  nam <- paste("is_", high_star_kw[i], sep = "")
  business_rest1000_review100_review[[nam]] = grepl(high_star_kw[i],business_rest1000_review100_review$text)
}
######## numerize dummy variables
for (i in 28:91){
  business_rest1000_review100_review[,i] = as.numeric(business_rest1000_review100_review[,i])
}

####### groupby and sum
business_rest1000_review100_review_catgo = business_rest1000_review100_review %>% group_by(business_id) %>% summarise(cnt=n()) 
attach(business_rest1000_review100_review)
####### groupby and sum
business_rest1000_review100_review_catgo = business_rest1000_review100_review %>% group_by(business_id) %>% summarise(sum_great=sum(is_great),sum_delicious=sum(is_delicious),sum_best=sum(is_best),sum_chicken=sum(is_chicken),sum_amazing=sum(is_amazing),sum_staff=sum(is_staff),sum_server=sum(is_server),sum_fresh=sum(is_fresh),sum_friendly=sum(is_friendly),sum_salad=sum(is_salad),sum_people=sum(is_people),sum_meat=sum(is_meat),sum_favourite=sum(is_favourite),sum_sweet=sum(is_sweet),sum_atmosphere=sum(is_atmosphere),sum_awesome=sum(is_awesome),sum_hot=sum(is_hot),sum_tasty=sum(is_tasty),sum_excellent=sum(is_excellent),sum_new=sum(is_new),sum_pork=sum(is_pork),sum_perfect=sum(is_perfect),sum_happy=sum(is_happy),sum_quality=sum(is_quality),sum_special=sum(is_special),sum_wife=sum(is_wife),sum_husband=sum(is_husband),sum_portions=sum(is_portions),sum_selection=sum(is_selection),sum_attentive=sum(is_attentive),sum_green=sum(is_green),sum_seating=sum(is_seating),sum_clean=sum(is_clean),sum_wonderful=sum(is_wonderful),sum_flavorful=sum(is_flavorful),sum_decent=sum(is_decent),sum_style=sum(is_style),sum_group=sum(is_group),sum_garlic=sum(is_garlic),sum_quickly=sum(is_quickly),sum_extremely=sum(is_extremely),sum_ramen=sum(is_ramen),sum_tender=sum(is_tender),sum_crispy=sum(is_crispy),sum_yummy=sum(is_yummy),sum_fan=sum(is_fan),sum_free=sum(is_free),sum_fantastic=sum(is_fantastic),sum_fun=sum(is_fun),sum_cool=sum(is_cool),sum_quick=sum(is_quick),sum_perfectly=sum(is_perfectly),sum_family=sum(is_family),sum_waiter=sum(is_waiter),sum_enjoyed=sum(is_enjoyed),sum_location=sum(is_location),sum_spicy=sum(is_spicy),sum_price=sum(is_price),sum_quite=sum(is_quite),sum_home=sum(is_home),sum_busy=sum(is_busy),sum_waitress=sum(is_waitress),sum_top=sum(is_top),sum_thai=sum(is_thai))              
write.csv(business_rest1000_review100_review_catgo,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_rest1000_review100_review_catgo.csv')
