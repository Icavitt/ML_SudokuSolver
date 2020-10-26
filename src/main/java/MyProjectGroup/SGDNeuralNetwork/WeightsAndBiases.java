package MyProjectGroup.SGDNeuralNetwork;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Ian on 1/31/2016.
 */
public class WeightsAndBiases {

    private INDArray[] biases;
    private INDArray[] weights;

    WeightsAndBiases(int[] sizes) {

        INDArray[] biases = new INDArray[sizes.length];
        int i = 0;
        for(INDArray nd : biases) {

            if(i == 0){
                //the first layer of biases is 0, as this is the input layer for the whole NN
                nd = Nd4j.zeros(sizes.length, 1);
            }
            else {
                //randn outputs an INDArray that samples a normally distributed gaussian function
                nd = Nd4j.randn(sizes[i],1);
            }
            ++i;
        }
        i = 0;
        INDArray[] weights = new INDArray[sizes.length];
        for(;i < sizes.length-1; ++i){
            //sizes[i] is the number of rows(# of input neurons) sizes[i+1] is the number of neurons in the next layer
            weights[i] = Nd4j.randn(sizes[i], sizes[i+1]);
        }
    }

    public WeightsAndBiases(INDArray[] biases, INDArray[] weights){
        this.biases = biases;
        this.weights = weights;
    }

    public INDArray[] getBiases() {
        return biases;
    }

    public INDArray[] getWeights() {
        return weights;
    }

}
