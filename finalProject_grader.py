"""FILE VERSION: 10/10/19
"""

import random
import traceback
import time
import json


studentName = "TestStudent"
inputFileName = 'finalProject.py'


#load problems
with open("testData.jsonlist",'r') as F:
	allThreads = [json.loads(l) for l in F.readlines()]

outFile = open("grade_"+studentName+".txt", 'w')

def prnt(S):
	global outFile
	outFile.write(str(S) + "\n")
	print(S)

"""This task is graded based on what fraction of the test set you get correct. If you get baselineCorrect, then you get 0. If you get maxCorrect, you get full credit. Extra credit is possible (see code for details).
"""
baselineCorrect = 0.8 #NOTE: These values may not be the same when we determine your grades!
maxCorrect = 1.0
fullCredit = 40
maxScore = 50

#load student file
try:
	F = open(inputFileName, 'r', encoding="utf-8")
	exec("".join(F.readlines()))
except Exception as e:
	prnt("Couldn't open or execute '" + inputFileName + "': " + str(traceback.format_exc()))
	prnt("FINAL SCORE: 0")
	outFile.close()
	exit()

penalty = 0
try:
	prnt("CALLING YOUR loadModel() FUNCTION")
	startTime = time.time()
	loadModel()
	endTime = time.time()
except Exception as e:
	endTime = time.time()
	prnt("\tError arose: " + str(traceback.format_exc()))
	prnt("\tNOTE: We won't penalize you directly for this, but this is likely to lead to exceptions later.")
if endTime - startTime > 300:
	prnt("Time to execute was " + str(int(endTime-startTime)) + " seconds; this is too long (-10 points)")
	penalty += 10

numCorrect = 0
numProblems = 0
for thread in allThreads:
	answers = [post['majority_type'] for post in thread['posts']]
	numProblems += len(answers)
	for post in thread['posts']:
		for lbl in ['majority_type', 'majority_link', 'annotations']:
			if lbl in post:
				del post[lbl]
	
	prnt("\n\nTESTING ON INPUT PROBLEM:")
	prnt("\t" + json.dumps(thread))
	prnt("CORRECT OUTPUT:")
	print("\t" + str(answers))
	prnt("YOUR OUTPUT:")
	try:
		startTime = time.time()
		result = classify(thread)
		prnt("\t" + str(result))
		endTime = time.time()
		
		#evaluate answer
		partial = 0
		if len(result) != len(answers):
			prnt("You don't have the same number of items in your list!")
			#give partial credit for number of answers the same
			for label in set(answers):
				if answers.count(label) == result.count(label):
					partial += answers.count(label)
			partial /= 4
			prnt("Giving you " + str(partial) + " points out of " + str(len(answers)))
			numCorrect += partial
		else:
			for i in range(len(result)):
				if result[i]==answers[i]:
					partial += 1
			prnt("Giving you " + str(partial) + " points out of " + str(len(answers)))
			numCorrect += partial
	except Exception as e:
		prnt("Marked as incorrect; there was an error while executing this problem: " + str(traceback.format_exc()))
percentCorrect = numCorrect*1.0/numProblems
points = min(maxScore, fullCredit * (percentCorrect - baselineCorrect) / (maxCorrect - baselineCorrect))
# print((percentCorrect - baselineCorrect) / (maxCorrect - baselineCorrect))
points = max(0, points)
prnt("\nYou got " + str(percentCorrect*100) + "% correct: +" + str(points) + "/" + str(fullCredit) + " points")
if penalty != 0:
	points -= penalty
	prnt("After penalties: " + str(points))

prnt("=============================")
prnt("=======  FINAL GRADE  =======")
prnt("=============================")
prnt(str(points) + " / 100")

outFile.close()