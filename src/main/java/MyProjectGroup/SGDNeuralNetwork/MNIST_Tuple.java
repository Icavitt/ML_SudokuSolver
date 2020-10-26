package MyProjectGroup.SGDNeuralNetwork;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Created by Ian on 2/10/2016.
 */
public class MNIST_Tuple implements Serializable{
    //a 784x1 indarry holding the pixel values of the mnist picture
    private INDArray pic;

    //a 10x1 indarray holding all 0s and 1 at the desired number
    private INDArray value;

    public MNIST_Tuple(){

    }

    public MNIST_Tuple(INDArray pic, INDArray value){
        this.pic = pic;
        this.value = value;
    }


    public INDArray getValue() {
        return value;
    }

    public void setValue(INDArray value) {
        this.value = value;
    }

    public INDArray getPic() {
        return pic;
    }

    public void setPic(INDArray pic) {
        this.pic = pic;
    }


}
