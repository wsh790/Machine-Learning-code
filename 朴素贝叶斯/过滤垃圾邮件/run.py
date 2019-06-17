# import re
# mySent='this book is the best book on python or M.L. I have ever laid eyes upon.'
# # print mySent.split()
#
# regEx=re.compile('\\W*')
# listOfTokens=regEx.split(mySent)
#
# print [tok.lower() for tok in listOfTokens if len(tok)>0]

import bayes
print bayes.spamTest()