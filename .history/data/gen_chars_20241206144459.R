df <- read.csv("clean_tract_v2.csv") # Your input data file with patient characteristics.


basechars <- function(x) {
	n = nrow(x)
	age65 = nrow(subset(x, age_current >=65))
	tmp = data.frame(n=n,
			   	     age65=age65,
			   	     age65p=age65/n*100,
			   	     sdi=median(x$sdi),
			   	     sdiqr=IQR(x$sdi),
			   	     adi=median(x$ADI_national_rank),
			   	     adiqr=IQR(x$ADI_national_rank))
	for (col in c("age_current", "Systolic_BP", 
				  "Heart_Rate", "Resp", "cci")) {
		colseries <- x[[col]]
		coln = mean(colseries)
		tmp[col] = coln
		tmp[paste(col, "sd", sep="_")] = sd(colseries)
	}

	for (col in c("privinsurance", "Medicare",
				  "Medicaid", "hfsystolic", "hfdiastolic",
				  "hfother",
				  "htn", "dmany", "cad",
				  "ckd", "cpd", "cvd", "afib",
				  "pvd", "loceuhfloor",
       			  "loceuhicu", "loceclhfloor",
        	      "loceclhicu", "specialcvd",
        		  "specialint")) {
		coln = sum(x[col])
		tmp[col] = coln
		tmp[paste(col, "p", sep="_")] = coln/n*100
	}

	x$special_or <- x$specialeme | x$specialoth
	tmp["specialor"] <- sum(x$special_or)
	tmp["specialor_p"] <- (sum(x$special_or)/ n)*100
	return(tmp)
}


basechars(df)
basechars(subset(df, white == 1))
basechars(subset(df, black == 1))
basechars(subset(df, other == 1))
basechars(subset(df, female == 1))
basechars(subset(df, female == 0))



#