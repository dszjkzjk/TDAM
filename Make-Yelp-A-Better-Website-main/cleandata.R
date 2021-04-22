library(tidyr)
library(dplyr)
business = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/yelp_business.csv')
#business = na.omit(business)
us_states =c('AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL',
            'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE',
            'NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','TN',
            'TX','UT','VT','VA','WA','WV','WI','WY')
ca_states =c("AB", "BC","LB", "MB", "NB", "NF", "NS", "NU", "NW", "ON", "PE", "QC", "SK", "YU")
library(dplyr) r
business=business %>% mutate(is_us = (state %in% us_states))
business=business %>% mutate(is_ca = (state %in% ca_states))
write.csv(business,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/new_business.csv',row.names = FALSE)

#business=business %>% mutate(is_us = ifelse(state %in% us_states, "US",ifelse(state %in% ca_states, "Canada", 'Others')))
#write.csv(business,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/new_business_catgo.csv',row.names = FALSE)                                                  

new_business = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/new_business.csv')

review_withoutext = read.csv('/Users/junkangzhang/Downloads/yelp_review.csv',colClasses = c(NA,NA,NA,NA,NA,'NULL',NA,NA,NA))

head(review_withoutext,10)
write.csv(review_withoutext,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/review_withouttext.csv')

################## remove text
review = read.csv('/Users/junkangzhang/Downloads/yelp_review.csv',nrows=100)

user = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp_sample/sample_yelp_user.csv')

################## clean review
review_withouttext = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/review_withouttext.csv')

review_withouttext$date = format(as.Date(review_withouttext$date, format="%Y-%m-%d"),"%Y")
sample = head(review_withouttext,100)
review_withouttext = review_withouttext %>% group_by(business_id,date) %>% summarise(review_count = n(), mean_star = mean(stars))
review_cleaned = review_withouttext

write.csv(review_cleaned,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/review_cleaned.csv')


review_withouttext$date = format(as.Date(review_withouttext$date, format="%Y-%m-%d"),"%Y")
sample = head(review_withouttext,100)
review_withouttext2 = review_withouttext %>% group_by(business_id) %>% summarise(review_count = n(), mean_star = mean(stars))
review_cleaned = review_withouttext

write.csv(review_cleaned,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/review_cleaned.csv')

#################### clean checkins

#################### clean business_attr
business_attr_picked = subset(business_attr, select=c(business_id,BusinessParking_garage,BusinessParking_street,BusinessParking_validated,BusinessParking_lot,BusinessParking_valet))

business_attr_picked = business_attr_picked %>% mutate(all_na = ((BusinessParking_validated=='Na')&(BusinessParking_validated=='Na')&(BusinessParking_garage=='Na')&(BusinessParking_lot=='Na')&(BusinessParking_street=='Na')&(BusinessParking_valet=='Na')))

business_attr_picked1 = business_attr_picked %>% filter(all_na == FALSE)

attach(business_attr_picked1)
#business_attr_picked[business_attr_picked == 'Na'] <- NA

business_attr_picked2 = business_attr_picked1 %>% mutate(is_parking = ((BusinessParking_validated=='True')|(BusinessParking_garage=='True')|(BusinessParking_lot=='True')|(BusinessParking_street=='True')|(BusinessParking_valet=='True')))

write.csv(business_attr_picked2,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_attr_picked2.csv')

################### clean business_hours
business_hours = read.csv('/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/yelp_business_hours.csv')

business_hours = business_hours %>% mutate(all_na = ((monday=='None')&(tuesday=='None')&(wednesday=='None')&(thursday=='None')&(friday=='None')&(saturday=='None')&(sunday=='None')))

business_hours = business_hours %>% filter(all_na == FALSE)

business_hours = business_hours %>% mutate(is_monday = (monday!='None'))
business_hours = business_hours %>% mutate(is_tuesday = (tuesday!='None'))
business_hours = business_hours %>% mutate(is_wednesday = (wednesday!='None'))
business_hours = business_hours %>% mutate(is_thursday = (thursday!='None'))
business_hours = business_hours %>% mutate(is_friday = (friday!='None'))
business_hours = business_hours %>% mutate(is_saturday = (saturday!='None'))
business_hours = business_hours %>% mutate(is_sunday = (sunday!='None'))


business_hours[,c(2:9)] = NULL

write.csv(business_hours,'/Users/junkangzhang/Downloads/MMA 2019/Courses/INSY662/group project/yelp/business_hours_cleaned.csv')

##################### 
read.csv()
