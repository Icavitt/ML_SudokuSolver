package MyProjectGroup.SGDNeuralNetwork;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;


import java.io.Serializable;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;
import static org.nd4j.linalg.util.NDArrayUtil.exp;

public class NeuralNet implements Serializable{
    //number of layers in the NN
    private int layers;
    //number of weights/biases represented(layers -1)
    private int connections;
    // size of each layer in the NN
    private int[] sizes;
    //each INDARAAY in biases is a column vector holding that layer's biases
    private INDArray[] biases;
    // Each INDARRAY in weights nxm size, n is the input layer, and m the output, every [n,m] represents the weight of that link
    private INDArray[] weights;


    public NeuralNet(int[] sizes)
    {
        this.layers = sizes.length;
        this.connections = layers-1;
        this.sizes = sizes;
        this.biases = new INDArray[sizes.length-1];
        this.weights = new INDArray[sizes.length-1];

        int i = 0;
//        for(INDArray nd : this.biases) {
//            if(i == 0){
//                //the first layer of biases is 0, as this is the input layer for the whole NN
//                nd = Nd4j.zeros(sizes[i], 1);
//            }
//            else {
//                //randn outputs an INDArray that samples a normally distributed gaussian function
//                nd = Nd4j.randn(sizes[i],1);
//            }
//            this.biases[i] = nd;
//            ++i;
//        }
//        i = 0;
        for(;i < sizes.length-1; ++i){
            //sizes[i] is the number of rows(# of input neurons) sizes[i+1] is the number of neurons in the next layer
//            this.weights[i] = Nd4j.randn(sizes[i], sizes[i+1]);
            this.weights[i] = Nd4j.randn(sizes[i+1], sizes[i]);
            this.biases[i] = Nd4j.randn(sizes[i+1],1);
        }
    }

    public NeuralNet(int[] sizes, int test){
        this.layers = sizes.length;
        this.connections = layers-1;
        this.sizes = sizes;
        this.biases = new INDArray[sizes.length-1];
        this.weights = new INDArray[sizes.length-1];
        for(int i = 0;i < sizes.length-1; ++i){
            double[] fw = new double[sizes[i+1]*sizes[i]];
            for(int j = 0; j != sizes[i+1]*sizes[i]; ++j ){
                fw[j] = 0.5;
            }
            INDArray weight = Nd4j.create(fw,new int[]{sizes[i+1],sizes[i]});
            this.weights[i] = Nd4j.zeros(sizes[i+1], sizes[i]).add(weight);
            double[] fb = new double[sizes[i+1]];
            for(int j = 0; j != sizes[i+1]; ++j ){
                fb[j] = 0.5;
            }
            INDArray bias = Nd4j.create(fb,new int[]{sizes[i+1],1});
            this.biases[i] = Nd4j.zeros(sizes[i+1],1).add(bias);
        }
    }

   public INDArray propagate(INDArray input)
    {
        int i = 0;
        while(i != connections)
        {
            input = sigmoid((weights[i].mmul(input)).add(biases[i]));
            ++i;
        }
        return input;
    }

    public int detectInteger(INDArray input){
        INDArray a = propagate(input);
        int i = 0;
        int max = 0;
        double max_d = 0;
        while(i != 10){
            double temp = a.getDouble(i,0);
            if(temp > max_d){
                max_d = temp;
                max = i;
            }
            ++i;
        }
        return max;
    }

    public boolean isPrinted(INDArray input){
        INDArray a = propagate(input);
        int i = 0;
        int max = 0;
        double max_d = 0;
        while(i != 2){
            double temp = a.getDouble(i,0);
            if(temp > max_d){
                max_d = temp;
                max = i;
            }
            ++i;
        }
        if(max == 0){
            return true;
        }
        else{
            return false;
        }
    }

    public void print()
    {
        int i = 0;
        System.out.println("weights");
        while(i < connections)
        {
//            System.out.println("outputting layer" + i + "biases then weights");
            //System.out.println(biases[i]);
            System.out.println(weights[i]);
            ++i;
        }
        System.out.println("now biases");
        i = 0;
        while(i < layers -1 ){
            System.out.println(biases[i]);
            ++i;
        }
    }

    public int getLayers() {
        return layers;
    }

    public void setLayers(int layers) {
        this.layers = layers;
    }

    public int[] getSizes() {
        return sizes;
    }

    public void setSizes(int[] sizes) {
        this.sizes = sizes;
    }

    public INDArray[] getBiases() {
        return biases;
    }

    public void setBiases(INDArray[] biases) {
        this.biases = biases;
    }

    public INDArray[] getWeights() {
        return weights;
    }

    public void setWeights(INDArray[] weights) {
        this.weights = weights;
    }

}
