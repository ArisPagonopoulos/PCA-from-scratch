import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
from PIL import Image
import matplotlib.pyplot as plt

def standarize(m):
    """Accepts a matrix m and returns the scaled matrix, the means and std array"""

    #we calculate the mean per feature
    mu = np.mean(m, axis = 1)
    #the standard deviation per feature
    s = np.std(m, axis = 1)
    #we standarize the matrix
    scaled = (m - mu)/s
    return scaled, mu, s

#we read the image
img = Image.open("c:\\users\\aris\\desktop\\Lenna.png")

#from RGB to grayscale
img_gray = img.convert("L")

#we convert the image to numpy array
img_array = np.array(img_gray)

scaled_img, means, stds = standarize(img_array)
#size of the data
n = img_array.shape[1]

#Covariance Matrix
covariance_matrix = (1/n)*np.matmul(scaled_img, np.transpose(scaled_img))

#Calculate the eighenvalues and eighenvectors
eigh_values, eigh_vectors = np.linalg.eigh(covariance_matrix)

#We calculate the total variance, it is the sum of the eighenvalues of the array
total_variance = np.sum(eigh_values)

#we create an array to store the variance captured by pc's
variances = []
fig = plt.figure()

for comp in range(1,21,2):
    #comp is the number of principal combonents
    sub = fig.add_subplot(2,5,comp//2+1)
    #we calculate the variance lost
    variance_lost = np.sum(eigh_values[:-comp])/total_variance
    variances.append(1-variance_lost)

    sub.set_title(f"PC:{comp}, Variance Lost:{100*variance_lost:.2f}%", fontsize = 10)

    #we get the base
    pca = eigh_vectors[:,-comp::]

    #we transform the observation
    transform = np.matmul(pca, np.matmul(np.transpose(pca), scaled_img))*stds + means
    sub.imshow(transform, cmap = "gray", interpolation =  "nearest")

plt.show()
plt.scatter(x = list(range(1,21,2)), y = variances, color = "black")
plt.plot(list(range(1, 21, 2)), variances, color = "green")
plt.xticks(np.arange(1, 21, 2))
plt.title("Scree Plot")
plt.show()