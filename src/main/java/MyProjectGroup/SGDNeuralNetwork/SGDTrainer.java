package MyProjectGroup.SGDNeuralNetwork;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

/**
 * Created by Ian on 1/31/2016.
 */
public class SGDTrainer {

    private NeuralNet nn;
    private MNIST_Loader ml;
    private MNIST_Tuple[] training_data;
    private MNIST_Tuple[] validation_data;

    public SGDTrainer(){};

    public SGDTrainer(NeuralNet nn)
    {
        this.nn = nn;
    }

    public void train(int batchSize, int trainingRuns, double learningRate, double lambda, boolean outputProgress ){
        //divides training data up, calls update on each batch
        int i = 0;
        while (i <= trainingRuns){
            Collections.shuffle(Arrays.asList(training_data));
            int batches = training_data.length / batchSize;
            MNIST_Tuple[][] mini_batches = new MNIST_Tuple[batches][batchSize];
            //the nested for loop sets each minibatch to the appropriate training_data
            for(int j = 0, l = 0; l != batches; j += batchSize, ++l){
                mini_batches[l] = new MNIST_Tuple[batchSize];
                for(int k = 0; k != batchSize; ++k){
                    mini_batches[l][k] = training_data[k+j];
                }
            }
            //the following for loop calls the mini_batch update on each mini_batch
            int z = 0;
            for(MNIST_Tuple[] mini_batch : mini_batches){
                //System.out.println("batch: " + z++ + " of" + (mini_batches.length-1) + " in batches");
                mini_batchUpdate(mini_batch, learningRate, lambda);
            }
            if(outputProgress){
                evaluate();
            }
            ++i;
        }
    }
    //this is where the error from packprop is summed for each layer then applied to the nn
    public void mini_batchUpdate(MNIST_Tuple[] mini_batch, double learningRate, double lambda){
        //initialize the ndarays that will hold the partial derivatives of the cost function, for each entry in each layer to 0
        INDArray[] delB = new INDArray[nn.getLayers()-1];
        INDArray[] delW = new INDArray[nn.getLayers()-1];
        int i = 0;
        for(INDArray b : nn.getBiases()){
            if(i == nn.getLayers()-1){
                break;
            }
            delB[i] = Nd4j.zeros(b.shape());
            ++i;
        }
        i = 0;
        for(INDArray w : nn.getWeights()){
            if(i == nn.getLayers()-1){
                break;
            }
           delW[i] =  Nd4j.zeros(w.shape());
            ++i;
        }
        //start performing back propagation and summing the cumulative gradients for the network
        int x = 0;
        for(MNIST_Tuple mt : mini_batch){
            //System.out.println("Tuple" + x++ + "of " + (mini_batch.length-1) + " in batch" );
            //back prop is returning incorrect shape for weights
            WeightsAndBiases wb = backProp(mt);
            INDArray[] delta_delB = wb.getBiases();
            INDArray[] delta_delW = wb.getWeights();
            //following 2 loops dont actually do anything
//            for(INDArray b : delB){
//                b = b.add(delta_delB[i]);
//                ++i;
//            }
//            i = 0;
//            for(INDArray w : delW){
//                w = w.add(delta_delW[i]);
//                ++i;
//            }
            for(i = 0; i != nn.getLayers()-1; ++i){
                delB[i] = delB[i].add(delta_delB[i]);
                delW[i] = delW[i].add(delta_delW[i]);
            }
        }
        //now using the summed gradients at every point calculate the new weights and biases
        INDArray[] nnWeights = nn.getWeights();
        INDArray[] nnBiases = nn.getBiases();
        i = 0;
        for(i = 0; i != nn.getLayers()-1; ++i){
            nnWeights[i] = nnWeights[i].muli(1-learningRate*lambda);
            nnWeights[i] = nnWeights[i].sub(delW[i].muli((learningRate/mini_batch.length)));
            nnBiases[i] = nnBiases[i].sub(delB[i].muli((learningRate/mini_batch.length)));
        }
    }

    public void evaluate(){
        INDArray[] results = new INDArray[validation_data.length];
        int i = 0;
        int correct = 0;
        for(MNIST_Tuple mt : validation_data){
            results[i] = nn.propagate(mt.getPic());
            //getting each value out of the output, check which is highest and compare index to validation_data's tuples index for max entry
            double maxOut = 0;
            int maxI = -1;
            double maxCheck = 0;
            int maxII = 0;
            for(int j = 0; j != 10; ++j){ //if i want a more extensible program need to grab the size of the last nn layer rather than hard code it
                double temp = results[i].getDouble(j,0);
                double temp2 = mt.getValue().getDouble(j,0);
                if(temp > maxOut){
                    maxOut = temp;
                    maxI = j;
                }
                if(temp2 > maxCheck){
                    maxCheck = temp2;
                    maxII = j;
                }
            }
            if(maxI == maxII){
                ++correct;
            }
        }
        System.out.println(correct + "/" + validation_data.length + " correct");
//        System.out.println("Enter S to save the network in this state, anything else to continue training");
//        Scanner scan = new Scanner(System.in);
//        String s = scan.next();
//        if(s.equals("s") || s.equals("S")){
//            saveNN(correct);
//        }
        saveNN(correct , validation_data.length);
    }

    public void evaluate(MNIST_Tuple[] validation_data){
        INDArray[] results = new INDArray[validation_data.length];
        int i = 0;
        int correct = 0;
        for(MNIST_Tuple mt : validation_data){
            results[i] = nn.propagate(mt.getPic());
            //getting each value out of the output, check which is highest and compare index to validation_data's tuples index for max entry
            double maxOut = 0;
            int maxI = -1;
            double maxCheck = 0;
            int maxII = 0;
            for(int j = 0; j != 1; ++j){ //if i want a more extensible program need to grab the size of the last nn layer rather than hard code it
                double temp = results[i].getDouble(j,0);
                double temp2 = mt.getValue().getDouble(j,0);
                if(temp > maxOut){
                    maxOut = temp;
                    maxI = j;
                }
                if(temp2 > maxCheck){
                    maxCheck = temp2;
                    maxII = j;
                }
            }
            if(maxI == maxII){
                ++correct;
            }
        }
        System.out.println(correct + "/" + validation_data.length + " correct");
    }

    public WeightsAndBiases backProp(MNIST_Tuple mt){
        //initialize the ndarays that will hold the gradients of the cost function, for each entry in each layer to 0
        INDArray[] delB = new INDArray[nn.getLayers()-1];
        INDArray[] delW = new INDArray[nn.getLayers()-1];
        int i = 0;
        for(INDArray b : nn.getBiases()){
            if(i == nn.getLayers()-1){
                break;
            }
            delB[i] = Nd4j.zeros(b.shape());
            ++i;
        }
        i = 0;
        for(INDArray w : nn.getWeights()){
            if(i == (nn.getLayers() -1)){
                break;
            }
            delW[i] =  Nd4j.zeros(w.shape());
            ++i;
        }
        //initialize a new array of ndarrays that will hold the activations at each layer starting with the input
        INDArray activation = mt.getPic();
        INDArray[] activations = new INDArray[nn.getLayers()];
        activations[0] = activation;
        //initialize a new array of ndarrays that will hold the z values(activation * weight +bias)
        INDArray[] zValues = new INDArray[nn.getLayers()-1];
        //propagate and record values as you go
        for(i = 0; i < nn.getLayers()-1; ++i){
            INDArray z = (nn.getWeights()[i].mmul(activation));
            z = z.add(nn.getBiases()[i]);
            zValues[i] = z;
            activation = sigmoid(z);
            activations[i+1] = activation;
        }
        //break following line up into multiple delta shouldn't be 0
        INDArray delta = cost_derivative(activations[nn.getLayers()-1], mt.getValue());
//        delta = delta.muli(sigmoid_prime(zValues[nn.getLayers()-2]));
        delB[nn.getLayers()-2] = delta;
        delW[nn.getLayers()-2] = delta.mmul(activations[nn.getLayers()-2].transpose());
        //zvalues indexing wrong...or maybe 2nd loop goes too many times...
        //cahnged the way the weights and biases are created, zvalues indexing now definitely off
        //cahnging the following loop from -2(was -1) to minus 3 but doing plus 1 to zvalues cause it has nothing at 0
        //everything needs to be refactored at some point soon...
        for(i = nn.getLayers()-3; i != -1; --i){
            INDArray z = zValues[i];
            INDArray sp = sigmoid_prime(z);
            delta = (nn.getWeights()[i+1].transpose().mmul(delta)).muli(sp);
            delB[i] = delta;
            delW[i] = delta.mmul(activations[i].transpose());
        }
        return new WeightsAndBiases(delB, delW);
    }

    public INDArray cost_derivative(INDArray output, INDArray value){
        //simply subtracts the y from the output of the network
        //this is technically the partial derivatives of the overall function
        return (output.sub(value));
    }

    public INDArray sigmoid_prime(INDArray z){
        INDArray ones = Nd4j.ones(z.shape());
        return sigmoid(z).muli(ones.sub(sigmoid(z)));
    }

    public void saveNN(int correct, int total){
        double percent = ((double)correct)/((double)total)*100.0;
        String fileName = percent + "% My Print NN.ser";
        try
        {
            FileOutputStream fileOut = new FileOutputStream(fileName);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(nn);
            out.close();
            fileOut.close();
        }catch(IOException i)
        {
            i.printStackTrace();
        }
    }

    public NeuralNet getNn() {
        return nn;
    }

    public void setNn(NeuralNet nn) {
        this.nn = nn;
    }

    public void setTraining_data(MNIST_Tuple[] training_data) {
        this.training_data = training_data;
    }

    public void setValidation_data(MNIST_Tuple[] validation_data) {
        this.validation_data = validation_data;
    }
}
