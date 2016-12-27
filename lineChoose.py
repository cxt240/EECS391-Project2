import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import random

# method to read the csv file into tuples
def read(file):
	with open(file, 'rt') as f:
		reader = csv.reader(f)
		data = []
		for row in reader:
			data.append(row)
		data.pop(0)
	return data


# method to extract the data of a single column according to the flower types
def classify(data, column, name):
	category = []
	for row in data:
		if row[4] == name:	
			category.append(row[column])

	return category


# plotting the scatterplot a certain color
def plot(x, y, specColor):
	plt.scatter(x, y, color=specColor)
	return


# scatterplot with two differ color points
def scatter(x1, y1, x2, y2):
	plot(x1, y1, 'blue')
	plot(x2, y2, 'red')
	return


# plotting a line
def line(m, b, color):
	x = np.arange(3.0, 7.0, 0.1)
	plt.plot(x, ((m * x) + b), color)
	return


# linear decision oboundary classification given two sets of x,y pair data and the slope and intercept of the decision line
def decision(x1, y1, x2, y2, m, b):
	categoryX1 = []
	categoryY1 = []
	categoryX2 = []
	categoryY2 = []

	# classifying the points according to where it is relative to the decision line
	for (x,y) in zip(x1, y1):
		if float(y) > ((m * float(x)) + b):
			categoryX1.append(x)
			categoryY1.append(y)
		else:
			categoryX2.append(x)
			categoryY2.append(y)

	for (x,y) in zip(x2, y2):
		if float(y) > ((m * float(x)) + b):
			categoryX1.append(x)
			categoryY1.append(y)
		else:
			categoryX2.append(x)
			categoryY2.append(y)

	# plotting the results and the decision line
	plot(categoryX1, categoryY1, 'blue')
	plot(categoryX2, categoryY2, 'red')
	line(m, b, 'black')
	return


# method that defines the MSE given the data, decision line and the two types of flower
def error(data, boundary, pattern):

	x1 = classify(data, 2, pattern[0])
	y1 = classify(data, 3, pattern[0])
	x2 = classify(data, 2, pattern[1])
	y2 = classify(data, 3, pattern[1])

	# computing the difference between the observed and expected squared
	points = 0
	error_sum = 0
	for (x,y) in zip(x1, y1):
		points = points + 1
		difference = ((float(boundary[0]) * float(x)) + float(boundary[1])) - float(y)
		error_sum = error_sum + math.pow(difference, 2)

	for (x,y) in zip(x2, y2):
		points = points + 1
		difference = ((float(boundary[0]) * float(x)) + float(boundary[1])) - float(y)
		error_sum = error_sum + math.pow(difference, 2)

	# calculating the error
	error = error_sum / points
	print("Mean squared error: ", error)
	return error


# the single step of a gradient
def gradient(data, boundary, pattern):

	x1 = classify(data, 2, pattern[0])
	y1 = classify(data, 3, pattern[0])
	x2 = classify(data, 2, pattern[1])
	y2 = classify(data, 3, pattern[1])

	# calculating the step of the gradient
	points = 0
	gradient_b = 0
	gradient = 0
	for (x,y) in zip(x1, y1):
		points = points + 1
		equation = float(boundary[0]) * float(x) + boundary[1]
		gradient_b = gradient_b + (equation - float(y))
		current = (equation - float(y)) * float(x)
		gradient = gradient + current

	for (x,y) in zip(x2, y2):
		points = points + 1
		equation = float(boundary[0]) * float(x) + boundary[1]
		gradient_b = gradient_b + (equation - float(y))
		current = (equation - float(y)) * float(x)
		gradient = gradient + current

	# calculating the actual step and multiplying it by the epsilon value to produce a new slope and intercept
	change = (gradient * 2) / points
	change_b = (gradient_b * 2) / points
	delta = 0.1 / points
	m = boundary[0] - change * delta
	b = boundary[1] - change_b * delta
	print('Gradient: Change from', boundary[0], boundary[1], 'to ', m, b)
	return m, b


# the gradient descent function
def descent(data, boundary, pattern):

	# instantiating values
	calc_error = 0
	prev_error = 100
	calc_error = error(data, boundary, pattern)
	m = boundary[0]
	b = boundary[1]

	# for plotting purposes
	iteration = 1
	trials = []
	offset = []

	slope = []
	intercept = []

	# gradient descent by calling the step function until the MSE doesn't change more than 0.00001 ("converges")
	while prev_error - calc_error > 0.00001:

		slope.append(m)
		intercept.append(b)
		trials.append(iteration)

		m, b = gradient(data, (m,b), pattern)

		iteration = iteration + 1
		offset.append(calc_error)
		prev_error = calc_error
		calc_error = error(data, (m, b), pattern)

	offset.append(calc_error)
	trials.append(iteration)
	slope.append(m)
	intercept.append(b)

	# for the scatterplot
	petal_len0 = classify(data, 2, 'virginica')
	petal_wid0 = classify(data, 3, 'virginica')
	petal_len1 = classify(data, 2, 'versicolor')
	petal_wid1 = classify(data, 3, 'versicolor')

	# middle plot
	scatter(petal_len0, petal_wid0, petal_len1, petal_wid1)
	middle = (len(trials) / 2) - (len(trials) % 2) * 0.5
	midSlope = slope[int(middle)]
	midInt = intercept[int(middle)]
	line(midSlope, midInt, 'black')
	label()
	plt.title("Middle Location of Gradient Descent")

	plt.subplot(222)
	plt.axis([-2000, 8000, 0, 1])
	mini_x = trials[0:int(middle)]
	mini_y = offset[0:int(middle)]
	plot(mini_x, mini_y, 'black')
	label2()
	plt.title("Middle MSE of Gradient Descent")
	print(midSlope, midInt)

	#final plot
	plt.figure(5)
	plt.subplot(221)
	scatter(petal_len0, petal_wid0, petal_len1, petal_wid1)
	line(slope[len(slope) - 1], intercept[len(intercept) - 1], 'black')
	label()
	plt.title("Final Location of Gradient Descent")

	plt.subplot(222)
	plot(trials, offset, 'black')
	label2()
	plt.title("Final MSE of Gradient Descent")
	return


# mean of a category
def center(x1, y1):

	points = 0
	xsum = 0
	ysum = 0
	for(x, y) in zip(x1, y1):
		xsum = xsum + float(x)
		ysum = ysum + float(y)
		points = points + 1

	centerx = xsum / points
	centery = ysum / points

	return centerx, centery


# Distance formula calculation given two points
def distance(x, y, x1, y1):

	x_meas = math.pow((float(x) - float(x1)), 2)
	y_meas = math.pow((float(y) - float(y1)), 2)
	dist = math.pow((x_meas + y_meas), 0.5)
	return dist

# used initially to find the radius of each decision circle
def findRad(x2, y2, x, y, x1, y1): 

	# finding the midpoint between the green and red
	mid1_x = (float(x2) + float(x)) / 2
	mid1_y = (float(y2) + float(y)) / 2

	# finding the midpoint between the red and the blue
	mid2_x = (float(x1) + float(x)) / 2
	mid2_y = (float(y1) + float(y)) / 2

	# finding the green and blue circle radii
	r1 = distance(x2, y2, mid1_x, mid1_y)
	r3 = distance(mid2_x, mid2_y, x1, y1)

	# determining the red circle radius (so it doesn't overlap
	r2 = 0
	if r1 < r3:
		r2 = r1
	else:
		r2 = r3

	return r1, r2, r3


# classifier for the circle decision boundaries. given all sets of data as well as the centers and radii of all circles
def circleDecision(center0, center1, center2, rad0, rad1, rad2, data0, data1, data2):

	x0 = float(center0[0])
	y0 = float(center0[1])
	x1 = float(center1[0])
	y1 = float(center1[1])
	x2 = float(center2[0])
	y2 = float(center2[1])

	for (x, y) in zip(data0[0], data0[1]):
		distance0 = distance(x, y, x0, y0)
		distance1 = distance(x, y, x1, y1)
		distance2 = distance(x, y, x2, y2)
		if distance0 <= rad0:
			plt.scatter(x, y, color = 'green', marker = 'x')
		elif distance1 <= rad1:
			plt.scatter(x, y, color = 'blue')
		elif distance2 <= rad2:
			plt.scatter(x, y, color = 'red', marker = 'x')
		else:
			plt.scatter(x, y, color = 'black', marker = 'x')

	for (x, y) in zip(data1[0], data1[1]):
		distance0 = distance(x, y, x0, y0)
		distance1 = distance(x, y, x1, y1)
		distance2 = distance(x, y, x2, y2)
		if distance0 <= rad0:
			plt.scatter(x, y, color = 'green', marker = 'x')
		elif distance1 <= rad1:
			plt.scatter(x, y, color = 'blue', marker = 'x')
		elif distance2 <= rad2:
			plt.scatter(x, y, color = 'red')
		else:
			plt.scatter(x, y, color = 'black', marker = 'x')

	for (x, y) in zip(data2[0], data2[1]):
		distance0 = distance(x, y, x0, y0)
		distance1 = distance(x, y, x1, y1)
		distance2 = distance(x, y, x2, y2)
		if distance0 <= rad0:
			plt.scatter(x, y, color = 'green')
		elif distance1 <= rad1:
			plt.scatter(x, y, color = 'blue', marker = 'x')
		elif distance2 <= rad2:
			plt.scatter(x, y, color = 'red', marker = 'x')
		else:
			plt.scatter(x, y, color = 'black', marker = 'x')
	return 


# Scatterplot label
def label():

	plt.xlabel('Petal Length')
	plt.ylabel('Petal Width')
	return


# MSE plot label
def label2():

	plt.xlabel('Iterations')
	plt.ylabel('Mean Squared Error')
	return


def main():
	# reading the csv file and classifying the data
	data = read('irisdata.csv')

	petal_len0 = classify(data, 2, 'virginica')
	petal_wid0 = classify(data, 3, 'virginica')

	petal_len1 = classify(data, 2, 'versicolor')
	petal_wid1 = classify(data, 3, 'versicolor')

	# Question 1 plots
	plt.figure(1)
	plt.subplot(221)
	scatter(petal_len0, petal_wid0, petal_len1, petal_wid1)
	label()
	plt.title('Scatterplot of petal length and petal width')

	plt.subplot(222)	
	scatter(petal_len0, petal_wid0, petal_len1, petal_wid1)
	label()
	line(-0.371, 3.4, 'black')
	plt.title('Arbitrary linear decision boundary')

	plt.subplot(223)
	label()
	decision(petal_len0, petal_wid0, petal_len1, petal_wid1, -0.371, 3.4)
	plt.title('Classifying based on the Linear boundary')

	# Question 2 plots
	plt.figure(2)
	plt.subplot(221)
	label()
	scatter(petal_len0, petal_wid0, petal_len1, petal_wid1)
	line(-0.371, 3.4, 'black')
	plt.title("Arbitrary Linear Decision Boundary and Scatterplot")

	# Small error boundary decision
	plt.subplot(223)
	decision(petal_len0, petal_wid0, petal_len1, petal_wid1, -0.371, 3.4)
	label()
	plt.title('Small error classification')

	# Large error decision
	plt.subplot(224)
	label()
	decision(petal_len0, petal_wid0, petal_len1, petal_wid1, -0.15, 1.5)
	plt.title('Large error classification')

	# output for the small and large MSE values
	print("Small error")
	error(data, (-0.371, 3.4), ('virginica', 'versicolor'))
	print("Large error")
	error(data, (-0.15, 1.5), ('virginica', 'versicolor'))

	# demonstration of the small step gradient
	plt.subplot(222)
	label()
	scatter(petal_len0, petal_wid0, petal_len1, petal_wid1)
	line(-0.371, 3.4, 'black')
	m, b = 	gradient(data, (-0.371, 3.4), ('virginica', 'versicolor'))
	line(m, b, 'purple')
	plt.title("Gradient Step Demo")

	# Question 3
	random.seed(240)
	b = random.uniform(0, 3)
	m_max = -b / 3
	m = random.uniform(0, m_max)

	# Initial plot
	plt.figure(3)
	plt.subplot(221)
	label()
	scatter(petal_len0, petal_wid0, petal_len1, petal_wid1)
	line(m, b, 'black')
	plt.title("Start Plot")

	# initial learning curve start	
	plt.subplot(222)
	x = error(data, (-0.15, 1.5), ('virginica', 'versicolor'))
	plot(0, x, 'black')	
	plt.axis([-2000, 10000, 0, 1])
	label2()
	plt.title("Start of Gradient Descent")

	# start of the middle plot
	plt.figure(4)
	plt.subplot(221)
	descent(data, (m, b), ('virginica', 'versicolor'))

	# Extra Credit question
	plt.figure(6)
	label()
	petal_len2 = classify(data, 2, 'setosa')
	petal_wid2 = classify(data, 3, 'setosa')

	# plotting circles
	virCircle = plt.Circle((6, 2.026), 1.1, color = 'blue', fill = False)
	verCircle = plt.Circle((4, 1.326), 1.01, color = 'red', fill = False)
	setCircle = plt.Circle((1.464, 0.244), 1, color = 'green', fill = False)

	# adding circles to the plot
	ax = plt.gca()
	ax.add_artist(virCircle)
	ax.add_artist(verCircle)
	ax.add_artist(setCircle)

	circleDecision((1.464, 0.244), (6, 2.026), (4, 1.326), 1, 1.15, 1.1, (petal_len0, petal_wid0), (petal_len1, petal_wid1), (petal_len2, petal_wid2))
	plt.title('Circle Decision and classification')

	plt.show()

	# for testing:
	# output for the small and large MSE values again
	print("Small error")
	error(data, (-0.371, 3.4), ('virginica', 'versicolor'))
	print("Large error")
	error(data, (-0.15, 1.5), ('virginica', 'versicolor'))
	return 


if __name__ == "__main__":
	main()
