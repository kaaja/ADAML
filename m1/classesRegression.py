from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
import seaborn
seaborn.set(style="white", context="notebook", font_scale=1.5,
            rc={"axes.grid": True, "legend.frameon": False,
"lines.markeredgewidth": 1.4, "lines.markersize": 10})
seaborn.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4.5})
from collections import OrderedDict
from scipy import linalg
from numpy.linalg import norm
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score


class LeastSquares:
    """ 
    Least squares and Ridge estimation of 2D function.
    Takes in meshgrid versions of x, y and z
    """
    
    def __init__(self, xPlot, yPlot, zPlot, degree, trueFunction=False, lambdaValue=None):
        if trueFunction:
            self.trueFunction = trueFunction
        else:
            self.trueFunction = self.frankeFunction
        if lambdaValue != None:
            self.lambdaValue = lambdaValue
        else:
            self.lambdaValue = 0.
        self.xOrg, self.yOrg = xPlot[0], yPlot[:,0]
        self.numberOfObservations = len(self.xOrg)
        self.xPlot, self.yPlot, self.zPlot = xPlot, yPlot, zPlot
        self.x, self.y, self.z = np.reshape(xPlot, -1, 1), np.reshape(yPlot, -1, 1), np.reshape(zPlot, -1, 1)
        self.degree = degree
        
    def createDesignMatrix(self, x=None, y=None):        
        if isinstance(x, np.ndarray):
            None          
        else:
            x, y = self.x, self.y
        self.XHat = np.c_[x, y] 
        poly = PolynomialFeatures(self.degree)
        self.XHat = poly.fit_transform(self.XHat)

    def estimate(self, z=0):
        if isinstance(z, np.ndarray):
            self.z = z

        XHat = self.XHat
        XHatTdotXHatShape = np.shape(XHat.T.dot(XHat))

        # SVD Ridge
        alphas = np.zeros(1)
        alphas[0] = self.lambdaValue
        U, s, Vt = linalg.svd(self.XHat, full_matrices=False)
        d = s / (s[:, np.newaxis].T ** 2 + alphas[:, np.newaxis])
        self.betaHat = np.dot(d * U.T.dot(self.z), Vt).T
        self.betaHat = np.squeeze(self.betaHat)       
    
    def predict(self):
        self.zPredict = self.XHat.dot(self.betaHat)

    def calculateResiduals(self):
        self.residuals = self.zPredict - self.z

        
    def plot(self, zPredict=0, xTest=None, yTest=None, zTest=None):
        if len(xTest) < 2:
            xPlot, yPlot, zPlot = self.xPlot, self.yPlot, self.zPlot
        else:
            xPlot, yPlot, zPlot = xTest, yTest, zTest
        try:
            zPredict = self.zPredict
        except:
            None
        zPredictPlot = (np.reshape(zPredict, np.shape(zPlot))).T
        
        
        # Plot
        fig = plt.figure()
        ax2 = fig.gca(projection='3d')
        surf = ax2.plot_surface(xPlot, yPlot, zPlot, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False) 
        ax2.zaxis.set_major_locator(LinearLocator(10))
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax2.set_title('True')
        plt.show()   

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xPlot, yPlot, zPredictPlot, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False) 
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_title('(Predicted/True-1)*100')
        plt.show()
        
             
        
        
    def calculateErrorScores(self):
        from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
        z, zPredict = self.z, self.zPredict
        self.mse = mean_squared_error(z, zPredict)
        self.r2 = r2_score(z, zPredict)
        

    def calculateVarianceBeta(self, noise=0, plotResiduals=False, nBins=50):
        XHat = self.XHat
        betaHat = self.betaHat
        z = self.z

        # Source Ridge estimator variance: https://arxiv.org/pdf/1509.09169.pdf
        shapeXXT = np.shape(XHat.T.dot(XHat))

        yFitted = self.zPredict
        mse = 1/(len(self.x) -1)*np.sum((z - yFitted)**2)
        if plotResiduals:
            self.calculateResiduals()
            plt.figure()
            n, bins, patches = plt.hist(self.residuals, nBins, density=True, \
            facecolor='green', alpha=0.75)
            plt.title('Residuals \n Noise: %.2f, Degree: %d' %(noise,self.degree))

        coefficientVariancesSLR = np.linalg.inv(XHat.T.dot(XHat)).dot(mse)
        W = np.linalg.inv(XHat.T.dot(XHat) + self.lambdaValue*np.eye(shapeXXT[0], shapeXXT[1])).dot(XHat.T).dot(XHat) 
        self.varBeta = W.dot(coefficientVariancesSLR).dot(W.T)
        self.varBeta = np.diag(self.varBeta)
        self.varOLS = np.diag(coefficientVariancesSLR)
        
    def movingAvg(self, array):
        n = len(array)
        cumsum  = np.cumsum(array)
        movingAverage  = np.cumsum(array)
        for counter in range(len(array)):
            movingAverage[counter] = cumsum[counter]/(counter+1)
        return movingAverage    
    
    def runningVariance(self, betaList):
        betaList = np.array(betaList)
        numberOfRuns, numberOfBetas = np.shape(betaList)
        varianceMatrix = np.zeros((numberOfRuns, numberOfBetas))

        for runNumber in range(numberOfRuns):
            for betaNumber in range(numberOfBetas):
                varianceMatrix[runNumber, betaNumber] = np.var(betaList[0:runNumber+1, betaNumber])
        return varianceMatrix
    
    def runningVarianceVector(self, betaList):
        betaList = np.array(betaList)
        numberOfRuns = len(betaList)
        varianceVector = np.zeros(numberOfRuns)

        for runNumber in range(numberOfRuns):
            varianceVector[runNumber] = np.var(betaList[0:runNumber+1])
        return varianceVector

    
    def errorBootstrap(self, bootstraps=2, plot=False, numberOfParamters=3): 
        
        self.mseBootstrap = np.zeros(bootstraps)
        self.R2Bootstrap = np.zeros(bootstraps)
        self.mseTraining = np.zeros(bootstraps)
        self.R2Training = np.zeros(bootstraps)

        
        self.bootstraps = bootstraps
        betaList = [] 
        self.error2 = np.zeros(bootstraps)
        residualDictUnknownF = OrderedDict()

        for i in range(self.numberOfObservations):
            for j in range(self.numberOfObservations):
                residualDictUnknownF[i,j] = []

        zPredictMeanMatrix = np.zeros((self.numberOfObservations, self.numberOfObservations))
        
		# Bias-variance decomposition when f is unknows
        mseMatrixUnknownF = np.zeros((self.numberOfObservations, self.numberOfObservations)) # Method
        sdMatrixUnknownF = np.zeros((self.numberOfObservations, self.numberOfObservations)) # Method
        bias2MatrixUnknownF = np.zeros((self.numberOfObservations, self.numberOfObservations)) # Method
        totalMatrixUnknownF = np.zeros((self.numberOfObservations, self.numberOfObservations)) # Method
        
        
        for iteration in range(self.bootstraps):
            # Training
            trainingIndices = [(np.random.randint(0, high=self.numberOfObservations), \
                      np.random.randint(0, high=self.numberOfObservations)) \
                     for i in range(self.numberOfObservations*self.numberOfObservations)]
            trainingIndices = np.array(trainingIndices)

            xTraining1D = self.xPlot[trainingIndices[:,1], trainingIndices[:,0]]
            yTraining1D = self.yPlot[trainingIndices[:,1], trainingIndices[:,0]]
            zTraining1D = self.zPlot[trainingIndices[:,1], trainingIndices[:,0]]
            
            self.x, self.y, self.z = xTraining1D, yTraining1D, zTraining1D        
            self.createDesignMatrix() # --> XHat
            self.estimate()           # --> betaHat
            betaList.append(self.betaHat)
            self.predict()
            self.mseTraining[iteration] = mean_squared_error(self.z, self.zPredict)
            self.R2Training[iteration] = r2_score(self.z, self.zPredict)
            
         
            # Testing
            testIndexArray = np.zeros((self.numberOfObservations, self.numberOfObservations))
            testIndexArray[trainingIndices[:,0], trainingIndices[:,1]] = 1
            testIndices = np.argwhere(testIndexArray == 0)
            
            xTest1D = self.xPlot[testIndices[:,1], testIndices[:,0]]
            yTest1D = self.yPlot[testIndices[:,1], testIndices[:,0]]
            zTest1D = self.zPlot[testIndices[:,1], testIndices[:,0]]
            
            self.x, self.y, self.z = xTest1D, yTest1D, zTest1D
            self.createDesignMatrix()
            self.predict()           # --> zPredict
            
            self.mseBootstrap[iteration] = mean_squared_error(self.z, self.zPredict)
            self.R2Bootstrap[iteration] = r2_score(self.z, self.zPredict)
            
            testIndicesTuple = tuple(map(tuple, testIndices))
            for coordinate, index in zip(testIndicesTuple, range(len(self.zPredict))):
                residualDictUnknownF[coordinate].append(self.z[index] - self.zPredict[index]) # unknown f
                
            
            self.error2[iteration] = 0            
            for i in range(len(self.x)):
                self.error2[iteration] += (self.z[i] - self.zPredict[i])**2
            
        for key, index in zip(residualDictUnknownF, range(len(residualDictUnknownF))):
			# Bias variance unknown f
            mseMatrixUnknownF[key[0], key[1]] = np.mean([(residualDictUnknownF[key][i])**2 for i in range(len(residualDictUnknownF[key]))])#np.mean( (residualDictUnknownF[key])**2 )
            sdMatrixUnknownF[key[0], key[1]] = np.var( residualDictUnknownF[key] )
            bias2MatrixUnknownF[key[0], key[1]] = ( np.mean(residualDictUnknownF[key]) )**2
            totalMatrixUnknownF[key[0], key[1]] = sdMatrixUnknownF[key[0], key[1]] + \
bias2MatrixUnknownF[key[0], key[1]]
         
        
		# Bias-variance average all positions
        self.mseUnknownF = np.nanmean(np.reshape(mseMatrixUnknownF, -1, 1))
        self.sdUnknownF = np.nanmean(np.reshape(sdMatrixUnknownF, -1, 1))
        self.bias2UnknownF = np.nanmean(np.reshape(bias2MatrixUnknownF, -1, 1))
        self.totalUnknownF = np.nanmean(np.reshape(totalMatrixUnknownF, -1, 1))
        
        self.error2 = np.sum(self.error2)
        self.mseBootStrapMA = self.movingAvg(self.mseBootstrap)
        self.R2BootstrapMA = self.movingAvg(self.R2Bootstrap)
        self.betaRunning = np.zeros((len(betaList[0]), len(betaList)))
        self.varianceBetaBootstrap = np.zeros((len(betaList[0]), len(betaList)))
        '''
        for i in range(len(betaList)):
            self.betaRunning[i, :] = self.movingAvg(betaList[i])
            self.varianceBetaBootstrap[i, :] = self.runningVariance(betaList[i])
        #self.betaRunning = self.movingAvg(betaList)
        
        self.varMSE = self.runningVarianceVector(self.mseBootstrap)
        '''
        self.varianceBetaBootstrap =  self.runningVariance(betaList)

        if plot:
            legends = []
            fig, ax = plt.subplots()
            
            for i in range(numberOfParamters):
                ax.plot(np.arange(1,np.shape(self.varianceBetaBootstrap)[0]+1), self.varianceBetaBootstrap[:,i])
                legends.append(r'$\beta_%d$' %i)
            ax.set_title('Bootstrap \n Running Var')
            ax.set_xlabel('Number of bootsraps')
            ax.set_ylabel('Running Var')
            fig.legend(legends)
            
            '''
            fig, ax = plt.subplots()
            ax.plot(np.arange(1,len(self.mseBootStrapMA)+1), self.mseBootStrapMA)
            ax.set_title('Bootstrap \n Running Mean MSE')
            ax.set_xlabel('Number of bootsraps')
            ax.set_ylabel('Running mean MSE')
            
            fig4, ax4 = plt.subplots()
            #ax4.plot(np.arange(1,len(self.mseBootStrapMA)+1), (np.sqrt(self.varMSE)/self.mseBootStrapMA)*100)
            ax4.plot(np.arange(1,len(self.mseBootStrapMA)+1), (np.sqrt(self.varMSE)))
            ax4.set_title('Bootstrap \n Running (Sd(MSE))')
            ax4.set_xlabel('Number of bootsraps')
            #ax4.set_ylabel(r'$Sd(\beta)$')
            
            fig2, ax2 = plt.subplots()
            ax2.plot(np.arange(1,len(self.R2BootstrapMA )+1), self.R2BootstrapMA )
            ax2.set_title('Bootstrap \n Running Mean R2')
            ax2.set_xlabel('Number of bootsraps')
            ax2.set_ylabel('Running Mean R2')
            
            fig3, ax3 = plt.subplots()
            ax3.plot(np.arange(1,len(self.R2BootstrapMA )+1), \
             np.sqrt(self.varianceBetaBootstrap)/self.betaRunning)
            ax3.set_title('Bootstrap \n Sd(\Beta)/Beta (Running)')
            ax3.set_xlabel('Number of bootsraps')
            ax3.set_ylabel(r'$Sd(\beta)$')
            '''
            


            
    def kFold(self, numberOfFolds=3, shuffle=False):
        self.betaList = [] # For variance calculation
        self.zPredictDict = OrderedDict() # Bias-variance decomposition
        self.mseTraining = np.zeros(numberOfFolds)
        self.R2Training = np.zeros(numberOfFolds)        
        noiseDict = OrderedDict()
        totalErrorDict = OrderedDict()
        varianceDict = OrderedDict()
        self.mseSciKit = np.zeros(numberOfFolds)
        self.R2SciKit = np.zeros(numberOfFolds)
        bias2Matrix = np.zeros((self.numberOfObservations, self.numberOfObservations))
        varianceMatrix = np.zeros_like(bias2Matrix)
        noiseMatrix = np.zeros_like(bias2Matrix)
        totalErrorMatrix = np.zeros_like(bias2Matrix)
        totalErrorMatrixForTesting = np.zeros_like(bias2Matrix)    

        errorMatrix = np.zeros_like(bias2Matrix) 
        
        for i in range(self.numberOfObservations):
            for j in range(self.numberOfObservations):
                self.zPredictDict[i, j]   = []
                noiseDict[i, j]      = []
                totalErrorDict[i, j] = []
        
        if not shuffle:
            indices = []
            for i in range(self.numberOfObservations):
                for j in range(self.numberOfObservations):
                    indices.append((i,j))
            indices = np.array(indices)
        else:
            return None
        
        foldLength = int(round(self.numberOfObservations**2/numberOfFolds))
        
        for iteration in range(numberOfFolds):
            if iteration != range(numberOfFolds)[-1]:
                testIndices = indices[iteration*foldLength:(iteration+1)*foldLength]
            else:
                testIndices = indices[foldLength*iteration:]   
            
            
            xTest1D = self.xPlot[testIndices[:,1], testIndices[:,0]]
            yTest1D = self.yPlot[testIndices[:,1], testIndices[:,0]]
            zTest1D = self.zPlot[testIndices[:,1], testIndices[:,0]]
            
            indices_rows = indices.view([('', indices.dtype)] * indices.shape[1])
            testIndices_rows = testIndices.view([('', testIndices.dtype)] * testIndices.shape[1])
            trainingIndices = np.setdiff1d(indices_rows, testIndices_rows).view(indices.dtype).reshape(-1, indices.shape[1])          
            xTraining1D = self.xPlot[trainingIndices[:,1], trainingIndices[:,0]]
            yTraining1D = self.yPlot[trainingIndices[:,1], trainingIndices[:,0]]
            zTraining1D = self.zPlot[trainingIndices[:,1], trainingIndices[:,0]]

            self.x, self.y, self.z = xTraining1D, yTraining1D, zTraining1D      
            
            self.createDesignMatrix() # --> XHat
            self.estimate()           # --> betaHat
            self.betaList.append(self.betaHat)
            self.predict()
            self.mseTraining[iteration] = mean_squared_error(self.z, self.zPredict)
            self.R2Training[iteration] = r2_score(self.z, self.zPredict)
            
            self.x, self.y, self.z = xTest1D, yTest1D, zTest1D
            self.createDesignMatrix()
            self.predict()           # --> zPredict
            self.mseSciKit[iteration] = mean_squared_error(self.z, self.zPredict)
            self.R2SciKit[iteration] = r2_score(self.z, self.zPredict)
            
            testIndicesTuple = tuple(map(tuple, testIndices))
            for coordinate, index in zip(testIndicesTuple, range(len(self.zPredict))):
                self.zPredictDict[coordinate].append(self.zPredict[index])
                noiseDict[coordinate].append((self.z[index] - self.trueFunction(self.x[index], self.y[index]))**2)
                totalErrorDict[coordinate].append((self.z[index] - self.zPredict[index])**2)
                errorMatrix[coordinate] = self.z[index] - self.zPredict[index]

        xForFunction, yForFunction = np.ravel(self.xPlot), np.ravel(self.yPlot)
        fValues = self.trueFunction(xForFunction, yForFunction)
        for key, index in zip(self.zPredictDict, range(len(self.zPredictDict))):
            zPredictMean = np.nanmean(self.zPredictDict[key])
            varianceMatrix[key[0], key[1]] = np.nanvar(self.zPredictDict[key])
            bias2Matrix[key[0], key[1]] = (zPredictMean - fValues[index])**2
            noiseMatrix[key[0], key[1]] = np.nanmean(noiseDict[key])
            totalErrorMatrix[key[0], key[1]] = np.nanmean(totalErrorDict[key])
            totalErrorMatrixForTesting[key[0], key[1]] = varianceMatrix[key[0], key[1]] + bias2Matrix[key[0], key[1]]\
                                                        + noiseMatrix[key[0], key[1]]
            
        # bias-variance over all observations
        self.bias2 = np.nanmean(np.reshape(bias2Matrix, -1, 1))
        self.variance = np.nanmean(np.reshape(varianceMatrix, -1, 1))
        self.noise = np.nanmean(np.reshape(noiseMatrix, -1, 1))
        self.totalError = np.nanmean(np.reshape(totalErrorMatrix, -1, 1))
        self.totalErrorForTesting = np.nanmean(np.reshape(totalErrorMatrixForTesting, -1, 1))
        
        # Bias-variance for cases when true function unknown, METHOD 1 (Baed on MSE's only)
        self.varianceMSERealData = np.var(self.mseSciKit)
        self.meanMSEsquaredRealData = (np.mean(self.mseSciKit))**2
        self.mseTotalRealData = norm(self.mseSciKit)
        self.mseTotalRealData = self.mseTotalRealData**2/numberOfFolds
        '''print('\n var+bias \n', self.varianceMSERealData + self.meanMSEsquaredRealData, \
             '\n mseTotal \n', self.mseTotalRealData )'''

        # Bias-variance, true function unknown, METHOD 2 all observations
        mseFromErrorMatrix = np.mean(errorMatrix**2)
        self.varianceMethod2 = np.sum( (self.zPredict - np.mean(self.zPredict))**2 )/len(self.zPredict)
        meanError = np.mean(errorMatrix)
        self.bias2Method2 =  np.sum( (np.reshape(self.zPlot, -1,1) - np.mean(self.zPredict))**2 )/len(self.zPredict) 
        self.mseMethod2 = np.mean(errorMatrix**2)
        self.mseMethod2SumBiasVariance = self.varianceMethod2 + self.bias2Method2

       
    def FrankeFunction(self, x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4 

################################################################################################

class Problem:
    def __init__(self, x, y, z, trueFunction=False):
        self.xPlot, self.yPlot, self.zPlot = x, y, z
        self.xPlotOrg, self.yPlotOrg, self.zPlotOrg = x, y, z
        if trueFunction:
            self.trueFunction = trueFunction
        else:
            self.trueFunction = self.frankeFunction

    def preditionAndPlot(self, xPlotTest, yPlotTest, zPlotTest, model='ridge', degree=5, lambdaValue=0, maxIterations=100000, plotResiduals=False, testPercentage=0, plot=False):
        xFlat = np.reshape(xPlotTest, -1, 1)
        yFlat = np.reshape(yPlotTest, -1, 1)
        zFlat = np.reshape(zPlotTest, -1, 1)
        IndexSplit = int(round(len(xFlat)*(1-testPercentage/100.)))
        
        xTrain = xFlat[0:IndexSplit]
        yTrain = yFlat[0:IndexSplit]
        zTrain = zFlat[0:IndexSplit]

        xTest = xFlat[IndexSplit:]
        yTest = yFlat[IndexSplit:]
        zTest = zFlat[IndexSplit:]


        if model=='ridge':
            ls = LeastSquares(self.xPlotOrg, self.yPlotOrg, self.zPlotOrg, degree=degree, trueFunction=self.trueFunction,\
                 lambdaValue=lambdaValue)
            ls.createDesignMatrix(xTrain, yTrain)

            ls.estimate(zTrain)

            ls.createDesignMatrix(xTest, yTest)

            ls.predict()
            ls.z = zTest

            ls.calculateErrorScores()
            self.mse = ls.mse
            self.r2 = ls.r2

            
            
        else:
            lasso=linear_model.Lasso(alpha=lambdaValue, fit_intercept=False, max_iter=maxIterations)
            polyLasso = PolynomialFeatures(degree)
            XHatLasso = np.c_[xTrain, yTrain] 
            XHatLasso = polyLasso.fit_transform(XHatLasso)
            lasso.fit(XHatLasso, zTrain)
            XHatLasso = np.c_[xTest, yTest] 
            XHatLasso = polyLasso.fit_transform(XHatLasso)
            zPredict = lasso.predict(XHatLasso)
            self.mse = mean_squared_error(zTest, zPredict)
            self.r2 = r2_score(zTest, zPredict)
            if plot:
                ls.plot(zPredict)

        
    def lsKfold(self, numberOfFolds=10, maxDegree=5):
        " Runs K-fold "
        self.maxDegree = maxDegree
        self.numberOfFolds = numberOfFolds
        self.degrees = np.arange(1, self.maxDegree+1)
        self.bootstraps = 100
        self.biasBS, self.varianceBS, self.noiseBS, self.totalErrorBS, self.mseBS, self.mseTrainingBS, \
        self.r2BS, self.biasPython , self.variancePython, self.mseAllBs, self.biasRealDataBS, \
        self.varianceRealDataBS, self.totalMSErealDataBS = \
        [], [], [], [], [], [], [], [], [], [], [], [], []
        self.biasKF, self.varianceKF, self.noiseKF, self.totalErrorKF, self.mseKF, self.mseTrainingKF, \
        self.r2KF, self.mseKFStd, self.biasRealDataKF, self.varianceRealDataKF, self.totalMSErealDataKF,\
        self.betasKF, self.varBetasKF, self.biasRealDataKF2, self.varianceRealDataKF2= \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        self.R2training, self.mseTraining, self.varBetasTraining, self.betasTraining = [], [], [], []

        for degree in self.degrees:
            ls = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree, trueFunction=self.trueFunction)
            ls.createDesignMatrix()
            ls.estimate()
            ls.predict()
            ls.calculateErrorScores()
            ls.calculateVarianceBeta()
            self.mseTraining.append(ls.mse)
            self.R2training.append(ls.r2)
            self.varBetasTraining.append(ls.varBeta)
            self.betasTraining.append(ls.betaHat)

            '''
            lsBS = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree, trueFunction=self.trueFunction)
            lsBS.errorBootstrap(bootstraps=self.bootstraps, plot=False)
            self.biasBS.append(lsBS.bias2)
            self.varianceBS.append(lsBS.variance)
            self.noiseBS.append(lsBS.noise)
            self.totalErrorBS.append(lsBS.totalError)
            self.mseBS.append(np.mean(lsBS.mseBootstrap))
            self.mseTrainingBS.append(np.mean(lsBS.mseTraining))
            self.r2BS.append(np.mean(lsBS.R2Bootstrap))
            self.biasPython.append(lsBS.biasPython)
            self.variancePython.append(lsBS.variancePython)
            self.biasRealDataBS.append(lsBS.meanMSEsquaredRealData)
            self.varianceRealDataBS.append(lsBS.varianceMSERealData)
            self.totalMSErealDataBS.append(lsBS.mseTotalRealData)
            '''
            lsKF = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree, trueFunction=self.trueFunction)
            lsKF.kFold(numberOfFolds=self.numberOfFolds)
            lsKF.calculateVarianceBeta()
            self.varBetasKF.append(lsKF.varBeta)
            self.biasKF.append(lsKF.bias2)
            self.varianceKF.append(lsKF.variance)
            self.noiseKF.append(lsKF.noise)
            self.totalErrorKF.append(lsKF.totalError)
            self.mseKF.append(np.mean(lsKF.mseSciKit))
            self.mseKFStd.append(np.std(lsKF.mseSciKit)/np.mean(lsKF.mseSciKit))
            self.mseTrainingKF.append(np.mean(lsKF.mseTraining))
            self.r2KF.append(np.mean(lsKF.R2SciKit))
            self.biasRealDataKF.append(lsKF.meanMSEsquaredRealData) # check
            self.biasRealDataKF2.append(lsKF.bias2Method2)
            self.varianceRealDataKF.append(lsKF.varianceMSERealData)
            self.varianceRealDataKF2.append(lsKF.varianceMethod2)
            self.totalMSErealDataKF.append(lsKF.mseTotalRealData)
            self.betasKF.append(lsKF.betaList)
            
    def varBeta(self, model='ols'):     
        "Compute parameter variance based on theoretical expression assuming\
         normally distributed error terms.\
         Plots histogram of variance for the three first coefficients."

        if model == 'ols':
            varBetasTrainingTobetasTraining = []
            for i in range(len(self.varBetasTraining)):
                varBetasTrainingTobetasTraining.append(self.varBetasTraining[i]/np.abs(self.betasTraining[i]))

            beta0 = np.zeros(len(self.varBetasTraining))
            beta1 = np.zeros(len(self.varBetasTraining))
            beta2 = np.zeros(len(self.varBetasTraining))

            for i in range(len(self.varBetasTraining)):
                beta0[i] = np.sqrt(varBetasTrainingTobetasTraining[i][0])
                beta1[i] = np.sqrt(varBetasTrainingTobetasTraining[i][1])
                beta2[i] = np.sqrt(varBetasTrainingTobetasTraining[i][2])
        
        elif model == 'ridge':
            varBetasTrainingTobetasTraining = [] 
            for i in range(len(self.varBetasTraining)):
                varBetasTrainingTobetasTraining.append(self.varBetasTrainingRidge[i]/np.abs(self.betasTrainingRidge[i]))

            beta0 = np.zeros(len(self.varBetasTrainingRidge))
            beta1 = np.zeros(len(self.varBetasTrainingRidge))
            beta2 = np.zeros(len(self.varBetasTrainingRidge))

            for i in range(len(self.varBetasTrainingRidge)):
                beta0[i] = np.sqrt(varBetasTrainingTobetasTraining[i][0])
                beta1[i] = np.sqrt(varBetasTrainingTobetasTraining[i][1])
                beta2[i] = np.sqrt(varBetasTrainingTobetasTraining[i][2])


        variableNames = [r'$\beta_0$', r'$\beta_1$', r'$\beta_2$' ]
        fig2,ax2 = plt.subplots(figsize=(8,4))
        numerOfVariables = 3
        ind = np.arange(numerOfVariables)  
        width = 0.15       # the width of the bars
        rects1 = ax2.bar(ind, np.asarray([beta0[0], beta1[0], beta2[0]]), width, color='g')
        rects2 = ax2.bar(ind+1*width, np.asarray([beta0[1], beta1[1], beta2[1]]), width, color='grey')
        rects3 = ax2.bar(ind+2*width, np.asarray([beta0[2], beta1[2], beta2[2]]), width, color='r')
        rects4 = ax2.bar(ind+3*width, np.asarray([beta0[3], beta1[3], beta2[3]]), width, color='b')
        rects5 = ax2.bar(ind+4*width, np.asarray([beta0[4], beta1[4], beta2[4]]), width, color='pink')


        fontSize = 20
        ax2.set_title(r'$Sd(\hat{\beta})/\hat{\beta},\;$ Noise term: %.2f' %self.noise)
        ax2.set_xticks((ind + width)*1.2 )#/ 2)
        ax2.set_xticklabels(variableNames)
        legends = [r'$Degree \; 1$',r'$Degree\;  2$' , r'$Degree\;  3$', r'$Degree \; 4$', r'$Degree \; 5$'   ]
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5)\
                   , fontsize = fontSize)


        
    def mseFigures(self, lastDegree=5):
        " Plots Training, testing MSE, R2"
        fig, (ax,ax2) = plt.subplots(1,2, figsize=(12.5,5))  # 1 row, 2 columns
        fontSize  = 16
        legends = ['K-fold', 'Training'] #, 'Bootstrap MSE', 'K-fold MSE'
        legendsR2 = ['K-fold', 'Training', '1']
        mseMethods =  self.mseKF, self.mseTraining
        r2Methods = self.r2KF , self.R2training
        xTicks = np.arange(1,self.maxDegree+1, 1)


        for mseMethod, r2Method, label in zip(mseMethods, r2Methods, legends): # 
            ax.plot(self.degrees[:lastDegree], mseMethod[:lastDegree])#, label=label)
            ax2.plot(self.degrees[:lastDegree], r2Method[:lastDegree])#, label=label)
        ax2.plot(self.degrees[:lastDegree], np.ones(len(self.degrees[:lastDegree])), 'k')


        ax.set_title('Average MSE', fontsize = fontSize*1.5)
        ax2.set_title('R2 score', fontsize = fontSize*1.5)
        ax.set_xlabel('Degrees of freedom', fontsize = fontSize*1.25)
        ax2.set_xlabel('Degrees of freedom', fontsize = fontSize*1.25)
        ax.set_xticks(xTicks)
        ax2.set_xticks(xTicks)
        ax2.set_ylim(0,1)
        ax2.set_yticks(np.arange(0,1+.1, .1))
        legendsAx2 = [r'$Bias^2$', 'Variance']
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1.0, box.height*.8])
        ax.legend(legends, loc='center left', bbox_to_anchor=(0.1, -0.55), \
                   fontsize = fontSize, ncol=2)
        ax.tick_params(axis='both', which='major', labelsize=fontSize*1.25)


        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 1.0, box.height*.8])
        ax2.legend(legendsR2, loc='center left', bbox_to_anchor=(0.2, -0.55), \
                   fontsize = fontSize, ncol=2)
        ax2.tick_params(axis='both', which='major', labelsize=fontSize*1.25)
        plt.tight_layout()

        legends = ['K-fold', 'Training'] #, 'Bootstrap MSE', 'K-fold MSE'
        
    def biasVariancePlot(self, lastDegree=5):
        r2Methods = self.r2KF , self.R2training
        fontSize = 15

        xTicks = np.arange(1,self.maxDegree+1, 1)

        fig, ax = plt.subplots(figsize=(8,4))

        ax.plot(self.degrees[:lastDegree], np.array(self.biasRealDataKF[:lastDegree])/np.array(self.totalMSErealDataKF[:lastDegree]))
        ax.plot(self.degrees[:lastDegree], np.array(self.varianceRealDataKF[:lastDegree])/np.array(self.totalMSErealDataKF[:lastDegree]))


        ax.set_title('Bias-variance decomposition \n K-fold', fontsize = fontSize*1.5)
        ax.set_xlabel('Degrees of freedom', fontsize = fontSize*1.25)
        ax.set_ylabel('Share of total MSE', fontsize = fontSize*1.25)
        ax.set_xticks(xTicks)
        legendsAx2 = [r'$Bias^2$', 'Variance', '1']
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1.0, box.height*.8])
        ax.legend(legendsAx2, loc='center left', bbox_to_anchor=(0.2, -0.55), \
                   fontsize = fontSize, ncol=2)
        ax.tick_params(axis='both', which='major', labelsize=fontSize*1.25)
        
    def biasVariance(self, mses):
        " Bias variance-decomposition \
        of MSE of MSEs"
        varianceMse = np.var(mses)
        meanMseSquared = (np.mean(mses))**2
        mseTotal = varianceMse + meanMseSquared
        return varianceMse, meanMseSquared, mseTotal
        
    def mseAllModels(self, noise=None, franke=False, maxDegree=5, numberOfFolds = 10, ridgeLambda = 1, lassoLambda = .001, maxIterations=10000, plotResiduals=False, nBins=50, residualsDegree=1):
        "Computes MSE train and test all models."
        self.ridgeLambda = ridgeLambda
        self.lassoLambda = lassoLambda
        self.noise = noise
        
        np.random.seed(1)

        if franke:
            if noise==None:
                noiseSize= 0
            else:
                noiseSize  = noise

            def frankeNoise(z, noiseSize):
                return z+ noiseSize*np.random.randn(len(z))

            self.zPlot = frankeNoise(self.zPlotOrg, noiseSize)
        self.maxDegree = maxDegree
        degrees =np.arange(1, self.maxDegree+1)
        bootstraps = 100

        self.biasBS, self.varianceBS, self.noiseBS, self.totalErrorBS, self.mseBS, self.mseTrainingBS,\
        self.r2BS, self.biasPython , self.variancePython, self.mseAllBs, \
        self.biasRealDataBS, self.varianceRealDataBS, self.totalMSErealDataBS = \
        [], [], [], [], [], [], [], [], [], [], [], [], []
        self.varBetasTraining = []

        self.biasKFLs, self.varianceKFLs, self.noiseKFLs, self.totalErrorKFLs, self.mseKFLs, self.mseTrainingKFLs, \
        self.r2KFLs, self.mseKFStdLs, self.biasRealDataKFLs, self.varianceRealDataKFLs, \
        self.totalMSErealDataKFLs, self.betasKFLs, self.varBetasKFLs, self.biasRealDataKFLs2, \
        self.varianceRealDataKFLs2 = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        self.biasKFRidge, self.varianceKFRidge, self.noiseKFRidge, self.totalErrorKFRidge, \
        self.mseKFRidge, self.mseTrainingKFRidge, self.r2KFRidge, self.mseKFStdRidge, \
        self.biasRealDataKFRidge, self.varianceRealDataKFRidge, self.totalMSErealDataKFRidge,\
        self.betasKFRidge, self.varBetasKFRidge, self.biasRealDataKFRidge2, \
         self.varianceRealDataKFRidge2= \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
        self.biasKFLasso, self.varianceKFLasso, self.noiseKFLasso, self.totalErrorKFLasso, self.mseKFLasso, \
        self.mseTrainingKFLasso, self.r2KFLasso, self.mseKFStdLasso, self.biasRealDataKFLasso, \
        self.varianceRealDataKFLasso, self.totalMSErealDataKFLasso, self.betasKFLasso, self.varBetasKFLasso, self.biasRealDataKFLasso2, self.varianceRealDataKFLasso2 = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        self.R2trainingLs, self.mseTrainingLs, self.varBetasTrainingLs, self.betasTrainingLs = [], [], [], []
        self.R2trainingRidge, self.mseTrainingRidge, self.varBetasTrainingRidge, self.betasTrainingRidge= [], [], [], []
        self.R2trainingLasso, self.mseTrainingLasso, self.varBetasTrainingLasso, self.betasTrainingLasso= [], [], [], []


        for degree in degrees:
            lsTrain = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree,trueFunction=self.trueFunction, lambdaValue=None)
            lsTrain.createDesignMatrix()
            lsTrain.estimate()
            lsTrain.predict()
            lsTrain.calculateErrorScores()
            if plotResiduals and degree == residualsDegree:
                lsTrain.calculateVarianceBeta(noise=self.noise, plotResiduals=plotResiduals,\
                                              nBins=nBins)
            else:
                lsTrain.calculateVarianceBeta(noise=self.noise)
            self.mseTrainingLs.append(lsTrain.mse)
            self.varBetasTraining.append(lsTrain.varBeta)
            self.R2trainingLs.append(lsTrain.r2)
            self.varBetasTrainingLs.append(lsTrain.varBeta)
            self.betasTrainingLs.append(lsTrain.betaHat)

            self.lsTrain = lsTrain

            ridgeTrain = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree,trueFunction=self.trueFunction, \
                                      lambdaValue=self.ridgeLambda)
            ridgeTrain.createDesignMatrix()
            ridgeTrain.estimate()
            ridgeTrain.predict()
            ridgeTrain.calculateErrorScores()
            ridgeTrain.calculateVarianceBeta()
            self.mseTrainingRidge.append(ridgeTrain.mse)
            self.R2trainingRidge.append(ridgeTrain.r2)
            self.varBetasTrainingRidge.append(ridgeTrain.varBeta)
            self.betasTrainingRidge.append(ridgeTrain.betaHat)

            lasso=linear_model.Lasso(alpha=lassoLambda, fit_intercept=False, max_iter=maxIterations)
            polyLasso = PolynomialFeatures(degree)
            XHatLasso = np.c_[np.reshape(self.xPlot, -1, 1), np.reshape(self.yPlot, -1, 1)] 
            XHatLasso = polyLasso.fit_transform(XHatLasso)
            lasso.fit(XHatLasso, np.reshape(self.zPlot, -1, 1))
            zPredictLasso = lasso.predict(XHatLasso)
            self.mseTrainingLasso.append(mean_squared_error(np.reshape(self.zPlot, -1, 1), zPredictLasso))


            lsKF = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree,trueFunction=self.trueFunction)
            lsKF.kFold(numberOfFolds=numberOfFolds)
            lsKF.calculateVarianceBeta()
            self.varBetasKFLs.append(lsKF.varBeta)
            self.biasKFLs.append(lsKF.bias2)
            self.varianceKFLs.append(lsKF.variance)
            self.noiseKFLs.append(lsKF.noise)
            self.totalErrorKFLs.append(lsKF.totalError)
            self.mseKFLs.append(np.mean(lsKF.mseSciKit))
            self.mseKFStdLs.append(np.std(lsKF.mseSciKit)/np.mean(lsKF.mseSciKit))
            self.mseTrainingKFLs.append(np.mean(lsKF.mseTraining))
            self.r2KFLs.append(np.mean(lsKF.R2SciKit))
            self.biasRealDataKFLs.append(lsKF.meanMSEsquaredRealData) # CHECK. Shouldnt it be bias?
            self.biasRealDataKFLs2.append(lsKF.bias2Method2)
            self.varianceRealDataKFLs.append(lsKF.varianceMSERealData)
            self.varianceRealDataKFLs2.append(lsKF.varianceMethod2)
            self.totalMSErealDataKFLs.append(lsKF.mseTotalRealData)
            self.betasKFLs.append(lsKF.betaList)

            ridgeKF = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree,trueFunction=self.trueFunction, lambdaValue = ridgeLambda)
            ridgeKF.kFold(numberOfFolds=numberOfFolds)
            ridgeKF.calculateVarianceBeta()
            self.varBetasKFRidge.append(ridgeKF.varBeta)
            self.biasKFRidge.append(ridgeKF.bias2)
            self.varianceKFRidge.append(ridgeKF.variance)
            self.noiseKFRidge.append(ridgeKF.noise)
            self.totalErrorKFRidge.append(ridgeKF.totalError)
            self.mseKFRidge.append(np.mean(ridgeKF.mseSciKit))
            self.mseKFStdRidge.append(np.std(ridgeKF.mseSciKit)/np.mean(ridgeKF.mseSciKit))
            self.mseTrainingKFRidge.append(np.mean(ridgeKF.mseTraining))
            self.r2KFRidge.append(np.mean(ridgeKF.R2SciKit))
            self.biasRealDataKFRidge.append(ridgeKF.meanMSEsquaredRealData) #check
            self.biasRealDataKFRidge2.append(ridgeKF.bias2Method2)
            self.varianceRealDataKFRidge.append(ridgeKF.varianceMSERealData)
            self.varianceRealDataKFRidge2.append(ridgeKF.varianceMethod2)
            self.totalMSErealDataKFRidge.append(ridgeKF.mseTotalRealData)
            self.betasKFRidge.append(ridgeKF.betaList)


            lassoKF = linear_model.Lasso(alpha=lassoLambda, fit_intercept=False, max_iter=maxIterations)
            lasso_scores = -cross_val_score(lassoKF, XHatLasso, np.reshape(self.zPlot, -1, 1),
                                     scoring="neg_mean_squared_error", cv=numberOfFolds)  
           
            self.mseKFLasso.append(np.mean(lasso_scores))

            varianceMseLasso, meanMseSquaredLasso, mseTotalLasso = self.biasVariance(lasso_scores)
            self.biasRealDataKFLasso.append(meanMseSquaredLasso)
            self.varianceRealDataKFLasso.append(varianceMseLasso)
            self.totalMSErealDataKFLasso.append(mseTotalLasso)


    def mseAllModelsPlot(self, lastDegree=5):
        "Plot of results from mseAllModels"
        # Plotting
        fig, (ax,ax2, ax3) = plt.subplots(1,3, figsize=(12.5,5))  # 1 row, 2 columns
        fontSize  = 16
        legends = ['OLS', r'Ridge $\lambda=$ %g' %self.ridgeLambda, 'Lasso']# r'Lasso $\lambda=$ %g' %lassoLambda] #, 'Bootstrap MSE', 'K-fold MSE'
        legends = ['OLS','Ridge', 'Lasso']

        mseMethods =  self.mseTrainingLs, self.mseTrainingRidge, self.mseTrainingLasso
        mseMethodsTesting =  self.mseKFLs, self.mseKFRidge, self.mseKFLasso
        biasVariance =  np.array(self.varianceRealDataKFLs)/np.array(self.totalMSErealDataKFLs), \
                        np.array(self.varianceRealDataKFRidge)/np.array(self.totalMSErealDataKFRidge),\
                        np.array(self.varianceRealDataKFLasso)/np.array(self.totalMSErealDataKFLasso)
        xTicks = np.arange(1,self.maxDegree+1, 1)

        for trainingMethod, testingMethod, bvModel, label in zip(mseMethods, mseMethodsTesting, biasVariance, legends): # mseTrainingBS, mseTrainingKF,
            ax.plot(self.degrees[:lastDegree], trainingMethod[:lastDegree])#, label=label)
            ax2.plot(self.degrees[:lastDegree], testingMethod[:lastDegree])#, label=label)
            ax3.plot(self.degrees[:lastDegree], bvModel[:lastDegree])


        ax.set_title('Training MSE ', fontsize = fontSize*1.25)
        ax.set_xlabel('Degrees of freedom', fontsize = fontSize*1.25)
        ax.set_xticks(xTicks)

        ax2.set_title('Mean testing MSE', fontsize = fontSize*1.25)
        ax2.set_xlabel('Degrees of freedom', fontsize = fontSize*1.25)
        ax2.set_xticks(xTicks)

        ax3.set_title('Variance share test MSE', fontsize = fontSize*1.25)
        ax3.set_xlabel('Degrees of freedom', fontsize = fontSize*1.25)
        ax3.set_xticks(xTicks)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1.0, box.height*.8])
        ax.legend(legends, loc='center left', bbox_to_anchor=(-0.1, -0.45), \
                   fontsize = fontSize, ncol=2)
        ax.tick_params(axis='both', which='major', labelsize=fontSize*1.25)


        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 1.0, box.height*.8])
        ax2.legend(legends, loc='center left', bbox_to_anchor=(-0.1, -0.45), \
                   fontsize = fontSize, ncol=2)
        ax2.tick_params(axis='both', which='major', labelsize=fontSize*1.25)
        #seaborn.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4.5})

        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 1.0, box.height*.8])
        ax3.legend(legends, loc='center left', bbox_to_anchor=(-0.1, -0.45), \
                   fontsize = fontSize, ncol=2)
        ax3.tick_params(axis='both', which='major', labelsize=fontSize*1.25)
        if self.noise !=None:
            fig.suptitle('Noise = %g' %self.noise, fontsize = fontSize*1.25)
            fig.subplots_adjust(top=0.88)#88
        else:
            plt.tight_layout()
        plt.tight_layout()
        
        
    def punishmenParameterAnalysis(self, degree=5, numberOfFolds = 10, startLambdaRidge = 0.05, \
                                   startLambdaLasso = 0.000125, numberOfPoints = 6,\
                                    maxIterations = 100000, adjustmentFactorRidge=1.5, \
                                    adjustmentFactorLasso=1.5):
        "Runs lambda and Ridge for different punishment parameters.\
         Plots the results"

        bootstraps = 100

        biasBS, varianceBS, noiseBS, totalErrorBS, mseBS, mseTrainingBS, r2BS, biasPython , variancePython, mseAllBs, \
        biasRealDataBS, varianceRealDataBS, totalMSErealDataBS = \
        [], [], [], [], [], [], [], [], [], [], [], [], []

        biasKFLs, varianceKFLs, noiseKFLs, totalErrorKFLs, mseKFLs, mseTrainingKFLs, \
        r2KFLs, mseKFStdLs, biasRealDataKFLs, varianceRealDataKFLs, totalMSErealDataKFLs, betasKFLs, varBetasKFLs, biasRealDataKFLs2, varianceRealDataKFLs2 = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        biasKFRidge, varianceKFRidge, noiseKFRidge, totalErrorKFRidge, mseKFRidge, mseTrainingKFRidge, \
        r2KFRidge, mseKFStdRidge, biasRealDataKFRidge, varianceRealDataKFRidge, totalMSErealDataKFRidge,\
        betasKFRidge, varBetasKFRidge, biasRealDataKFRidge2, varianceRealDataKFRidge2= \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        biasKFLasso, varianceKFLasso, noiseKFLasso, totalErrorKFLasso, mseKFLasso, mseTrainingKFLasso, \
        r2KFLasso, mseKFStdLasso, biasRealDataKFLasso, varianceRealDataKFLasso, totalMSErealDataKFLasso,\
        betasKFLasso, varBetasKFLasso= \
        [], [], [], [], [], [], [], [], [], [], [], [], []

        R2trainingLs, mseTrainingLs, varBetasTrainingLs, betasTrainingLs = [], [], [], []
        R2trainingRidge, mseTrainingRidge, varBetasTrainingRidge, betasTrainingRidge= [], [], [], []
        R2trainingLasso, mseTrainingLasso, varBetasTrainingLasso, betasTrainingLasso= [], [], [], []



        ridgeLambdas = np.array([startLambdaRidge*adjustmentFactorRidge**i for i in range(numberOfPoints)]) #np.arange(0.0025, 0.015+ 0.0025, 0.0025)
        lassoLambdas = np.array([startLambdaLasso*adjustmentFactorLasso**i for i in range(numberOfPoints)]) #np.arange(0.0025, 0.015+ 0.0025, 0.0025)


        for ridgeLambda, lassoLambda in zip(ridgeLambdas, lassoLambdas):
            lsTrain = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree,trueFunction=self.trueFunction, lambdaValue=None)
            lsTrain.createDesignMatrix()
            lsTrain.estimate()
            lsTrain.predict()
            lsTrain.calculateErrorScores()
            lsTrain.calculateVarianceBeta()
            mseTrainingLs.append(lsTrain.mse)
            R2trainingLs.append(lsTrain.r2)
            varBetasTrainingLs.append(lsTrain.varBeta)
            betasTrainingLs.append(lsTrain.betaHat)

            ridgeTrain = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree,trueFunction=self.trueFunction, lambdaValue=ridgeLambda)
            ridgeTrain.createDesignMatrix()
            ridgeTrain.estimate()
            ridgeTrain.predict()
            ridgeTrain.calculateErrorScores()
            ridgeTrain.calculateVarianceBeta()
            mseTrainingRidge.append(ridgeTrain.mse)
            R2trainingRidge.append(ridgeTrain.r2)
            varBetasTrainingRidge.append(ridgeTrain.varBeta)
            betasTrainingRidge.append(ridgeTrain.betaHat)

            lasso=linear_model.Lasso(alpha=lassoLambda, fit_intercept=False, max_iter=maxIterations)
            polyLasso = PolynomialFeatures(degree)
            XHatLasso = np.c_[np.reshape(self.xPlot, -1, 1), np.reshape(self.yPlot, -1, 1)] 
            XHatLasso = polyLasso.fit_transform(XHatLasso)
            lasso.fit(XHatLasso, np.reshape(self.zPlot, -1, 1))
            zPredictLasso = lasso.predict(XHatLasso)
            mseTrainingLasso.append(mean_squared_error(np.reshape(self.zPlot, -1, 1), zPredictLasso))


            lsKF = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree,trueFunction=self.trueFunction)
            lsKF.kFold(numberOfFolds=numberOfFolds)
            lsKF.calculateVarianceBeta()
            varBetasKFLs.append(lsKF.varBeta)
            biasKFLs.append(lsKF.bias2)
            varianceKFLs.append(lsKF.variance)
            noiseKFLs.append(lsKF.noise)
            totalErrorKFLs.append(lsKF.totalError)
            mseKFLs.append(np.mean(lsKF.mseSciKit))
            mseKFStdLs.append(np.std(lsKF.mseSciKit)/np.mean(lsKF.mseSciKit))
            mseTrainingKFLs.append(np.mean(lsKF.mseTraining))
            r2KFLs.append(np.mean(lsKF.R2SciKit))
            biasRealDataKFLs.append(lsKF.meanMSEsquaredRealData)
            biasRealDataKFLs2.append(lsKF.bias2Method2)
            varianceRealDataKFLs.append(lsKF.varianceMSERealData)
            varianceRealDataKFLs2.append(lsKF.varianceMethod2)
            totalMSErealDataKFLs.append(lsKF.mseTotalRealData)
            betasKFLs.append(lsKF.betaList)

            ridgeKF = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree,trueFunction=self.trueFunction, lambdaValue = ridgeLambda)
            ridgeKF.kFold(numberOfFolds=numberOfFolds)
            ridgeKF.calculateVarianceBeta()
            varBetasKFRidge.append(ridgeKF.varBeta)
            biasKFRidge.append(ridgeKF.bias2)
            varianceKFRidge.append(ridgeKF.variance)
            noiseKFRidge.append(ridgeKF.noise)
            totalErrorKFRidge.append(ridgeKF.totalError)
            mseKFRidge.append(np.mean(ridgeKF.mseSciKit))
            mseKFStdRidge.append(np.std(ridgeKF.mseSciKit)/np.mean(ridgeKF.mseSciKit))
            mseTrainingKFRidge.append(np.mean(ridgeKF.mseTraining))
            r2KFRidge.append(np.mean(ridgeKF.R2SciKit))
            biasRealDataKFRidge.append(ridgeKF.meanMSEsquaredRealData)
            biasRealDataKFRidge2.append(ridgeKF.bias2Method2)
            varianceRealDataKFRidge.append(ridgeKF.varianceMSERealData)
            varianceRealDataKFRidge2.append(ridgeKF.varianceMethod2)
            totalMSErealDataKFRidge.append(ridgeKF.mseTotalRealData)
            betasKFRidge.append(ridgeKF.betaList)


            lassoKF = linear_model.Lasso(alpha=lassoLambda, fit_intercept=False, max_iter=maxIterations)
            lasso_scores = -cross_val_score(lassoKF, XHatLasso, np.reshape(self.zPlot, -1, 1),
                                     scoring="neg_mean_squared_error", cv=numberOfFolds)  

            mseKFLasso.append(np.mean(lasso_scores))

            varianceMseLasso, meanMseSquaredLasso, mseTotalLasso = self.biasVariance(lasso_scores)
            biasRealDataKFLasso.append(meanMseSquaredLasso)
            varianceRealDataKFLasso.append(varianceMseLasso)
            totalMSErealDataKFLasso.append(mseTotalLasso)

            
        # Plotting
        fontSize  = 16
        legends = ['OLS', r'Ridge $\lambda=$ %g' %ridgeLambda, 'Lasso']# r'Lasso $\lambda=$ %g' %lassoLambda] #, 'Bootstrap MSE', 'K-fold MSE'
        legends = ['OLS','Ridge', 'Lasso']

        mseMethods =  mseTrainingLs, mseTrainingRidge, mseTrainingLasso
        mseMethodsTesting =  mseKFLs, mseKFRidge, mseKFLasso
        biasVariance =  np.array(varianceRealDataKFLs)/np.array(totalMSErealDataKFLs), \
                        np.array(varianceRealDataKFRidge)/np.array(totalMSErealDataKFRidge),\
                        np.array(varianceRealDataKFLasso)/np.array(totalMSErealDataKFLasso)
        xTicks = np.arange(1,self.maxDegree+1, 1)
        lastDegree = 5

        lambdaForPlot = [ridgeLambdas, lassoLambdas]
        modelNames = ['Ridge', 'Lasso']
        
        fig, (ax,ax2) = plt.subplots(1,2, figsize=(12.5,5))  # 1 row, 2 columns
        ax.plot(np.log2(ridgeLambdas), mseKFRidge)
        ax2.plot(np.log2(lassoLambdas), mseKFLasso)

        ax.set_title(modelNames[0]+'\n Test MSE', fontsize = fontSize*1.5)
        ax.set_xlabel(r'$\log(\lambda)$', fontsize = fontSize*1.25)
        ax.set_xticks(np.log2(lambdaForPlot[0]))

        ax2.set_title(modelNames[1]+'\n Test MSE', fontsize = fontSize*1.5)
        ax2.set_xlabel(r'$\log(\lambda)$', fontsize = fontSize*1.25)
        ax2.set_xticks(np.log2(lambdaForPlot[1]))

 
    def FrankeFunction(self, x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def varBetaFigures(self, noises=[.0], model='ols', lassoLambda=False, ridgeLambda=0):
        for noise in noises:
            self.mseAllModels(noise, franke=True, ridgeLambda=ridgeLambda)
            self.varBeta(model)

    def createResidualHistograms(self, noises=[0.], nBins=25,  residualsDegree=1):
        for noise in noises:
            self.mseAllModels(noise=noise, franke=True, plotResiduals=True, nBins=nBins, residualsDegree=residualsDegree)

    def bootstrapParameterVariance(self, bootstraps=100, plot=False):
        self.lsTrain.errorBootstrap(bootstraps=bootstraps, plot=plot)

    def varianceBiasDecompositionBootsrap(self, maxDegree = 25, bootstraps=100):
        " Runs the full bias variance decomposition using the Bootstrap method"
        degrees = np.arange(1,maxDegree)
        mseUnknownF = np.zeros(len(degrees))
        sdUnknownF = np.zeros(len(degrees))
        bias2UnknownF = np.zeros(len(degrees))
        totalUnknownF = np.zeros(len(degrees))

        counter = 0
        for degree in degrees:
            tst1 = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree, trueFunction=self.FrankeFunction)
            tst1.errorBootstrap(bootstraps=100)
            mseUnknownF[counter] = tst1.mseUnknownF
            sdUnknownF[counter] = tst1.sdUnknownF
            bias2UnknownF[counter] = tst1.bias2UnknownF
            totalUnknownF[counter] = tst1.totalUnknownF
            counter += 1
     
        fig, (ax, ax2) = plt.subplots(1,2,  sharex=True)
        fontSize  = 15
        #fig.suptitle('Bias-variance\ndecomposition')#, fontsize=fontSize)
        #fig.subplots_adjust(top=0.88)
        ax.set_title('Pure values')
        ax2.set_title('Share of total MSE')
        ax.plot(degrees, bias2UnknownF)
        ax.plot(degrees, sdUnknownF)
        ax.plot(degrees, totalUnknownF)
        ax2.plot(degrees, bias2UnknownF/totalUnknownF)
        ax2.plot(degrees, sdUnknownF/totalUnknownF)
        xTicks = np.arange(5,maxDegree+1, 5)

        ax.set_xlabel('Degrees of freedom', fontsize = fontSize*1.)
        ax2.set_xlabel('Degrees of freedom', fontsize = fontSize*1.)
        
        ax.set_xticks(xTicks)
        ax2.set_xticks(xTicks)
        legends = ['bias', 'variance', 'total']#, 'MSE scikit']
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(legends, loc='center left', bbox_to_anchor=(0, -0.35)\
           , fontsize = fontSize)
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.legend(legends, loc='center left', bbox_to_anchor=(0, -0.3)\
           , fontsize = fontSize)
        #plt.tight_layout()
        #plt.show()

    def betaVariancePlotBootstrap(self, degree = 5, bootstraps=100, numberOfParamters=3):
        tst1 = LeastSquares(self.xPlot, self.yPlot, self.zPlot, degree, trueFunction=self.FrankeFunction)
        tst1.errorBootstrap(bootstraps=bootstraps, plot=True)


def createFrankeData(noise=False, plot=False, observations=20):
    def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def frankeNoise(z, noiseSize):
        return z+ noiseSize*np.random.randn(len(z))
    
    np.random.seed(1)
    observationNumber = observations
    x = np.random.rand(observationNumber, 1)
    x = np.sort(x, 0)
    y = np.random.rand(observationNumber, 1)
    y = np.sort(y, 0)
    xPlot, yPlot = np.meshgrid(x,y)
    zPlot = FrankeFunction(xPlot, yPlot)
    if noise == False:
        noiseSize  = 0.
    else:
        noiseSize = noise
    zPlot = frankeNoise(zPlot, noiseSize)
    
    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xPlot, yPlot, zPlot, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title('Franke-function')
        plt.show()
    return xPlot, yPlot, zPlot



