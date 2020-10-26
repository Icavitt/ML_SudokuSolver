package MyProjectGroup.SGDNeuralNetwork;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
/**
 * Created by Ian on 2/11/2016.
 */
public class MNIST_Loader implements Serializable{
    //The training data is going to be split up between a training and validation set
    private MNIST_Tuple[] training_data;
    private MNIST_Tuple[] validation_data;

    private File images;// where I'm storing the MNIST training images "..\\..\\MNIST_Data\\train-images.idx3-ubyte";
    private File labels; // where I'm storing the MNIST training image labels "..\\..\\MNIST_Data\\train-images.idx3-ubyte";

    private int size = 60000;
    private int trainSize = 50000;
    private int validSize = 10000;
    public MNIST_Loader(){
        training_data = new MNIST_Tuple[50000];
        validation_data = new MNIST_Tuple[10000];
        for(int i = 0; i != size; ++i){
            if(i <= trainSize-1){
                training_data[i] = new MNIST_Tuple();
            }else{
                validation_data[i - trainSize] = new MNIST_Tuple();
            }
        }
        images = new File("..\\..\\MNIST_Data\\train-images.txt");
        labels = new File("..\\..\\MNIST_Data\\train-labels.txt");

//        labels = new File("..\\..\\MNIST_Data\\train-labels.idx1-ubyte");
//        images = new File("..\\..\\MNIST_Data\\train-images.idx3-ubyte");
    }

    public MNIST_Loader(int size){
        this.size = size;
        validSize = size / 10;
        trainSize = size - validSize;
        training_data = new MNIST_Tuple[size - validSize];
        validation_data = new MNIST_Tuple[validSize];
        for(int i = 0; i != size; ++i){
            if(i <= trainSize-1){
                training_data[i] = new MNIST_Tuple();
            }else{
                validation_data[i - trainSize] = new MNIST_Tuple();
            }
        }
        images = new File("my-printed-data.txt");
        labels = new File("my-printed-labels.txt");
    }

    public void load_data(){
        InputStream is;
        Reader r;
        BufferedReader brImages = null;
        BufferedReader brLabels = null;
        //first create the buffered readers for each file
        try{
            is = new FileInputStream(images);
            r = new InputStreamReader(is, "UTF8");
            brImages = new BufferedReader(r);
        }catch(IOException ioe){
            System.out.println("reader for images failed to be created");
        }
        try{
            is = new FileInputStream(labels);
            r = new InputStreamReader(is, "UTF8");
            brLabels = new BufferedReader(r);
        }catch (IOException ioe){
            System.out.println("reader for labels failed to be created");
        }
        try{
            readImages(brImages);
            readLabels(brLabels);
            MNIST_Tuple[] allData = new MNIST_Tuple[size];
            for(int i = 0; i != size; ++i){
                if(i <= trainSize-1){
                    allData[i] = training_data[i];
                }else{
                    allData[i] = validation_data[i - trainSize];
                }
            }
            Collections.shuffle(Arrays.asList(allData));
            Collections.shuffle(Arrays.asList(allData));
            for(int i = 0; i != size; ++i){
                if(i <= trainSize-1){
                    training_data[i] = allData[i];
                }else{
                    validation_data[i - trainSize] = allData[i];
                }
            }
        }catch(Exception e){
            System.out.println(e);
        }
    }
    public void readImages(BufferedReader br) throws IOException{
        if(Integer.parseInt(readFourBytes(br), 16) != 2051){
            System.out.println("first 4 bytes didn't match in the labels file");
            return;
        }
        if(Integer.parseInt(readFourBytes(br), 16) != size){
            System.out.println("second 4 bytes didn't match in the labels file");
//            return;
        }
        int rows = Integer.parseInt(readFourBytes(br), 16);
        int cols = Integer.parseInt(readFourBytes(br), 16);
        if(rows != cols){
            System.out.println("Rows not equal to columns, invalid pic format");
        }
        for(int i = 0; i != size; ++i){
            int ii = 0;
            double[] f = new double[rows*cols];
            for(int j = 0; j != rows; ++j){
                for(int k = 0; k != cols; ++k,++ii){
                    int temp = Integer.parseInt(readByte(br), 16);
                    double temp_d = temp;
                    f[ii] = temp_d/256;
                }
            }
            INDArray pic = Nd4j.create(f, new int[]{rows * cols, 1});
            ii = 0;
            if(i < trainSize){
                training_data[i].setPic(pic);
            }else{
                validation_data[i - trainSize].setPic(pic);
            }
        }
    }
    public void readLabels(BufferedReader br) throws IOException{
        if(Integer.parseInt(readFourBytes(br), 16) != 2049){
            System.out.println("first 4 bytes didn't match in the labels file");
            return;
        }
        if(Integer.parseInt(readFourBytes(br), 16) != size){
            System.out.println("second 4 bytes didn't match in the labels file");
            return;
        }
        for(int i = 0; i != size; ++i){
            int label = Integer.parseInt(readByte(br), 16);
            INDArray input = putInMatrix(label);
            if(i < trainSize){
                training_data[i].setValue(input);
            }else{
                validation_data[i - trainSize].setValue(input);
            }
            //System.out.println(input);
        }
    }

    public INDArray putInMatrix(int value){
        int i = 0;
        double[] f = new double[10];
        while(i != 10){
            if(i == value){
                f[i] = 1;
            }else{
                f[i] = 0;
            }
            ++i;
        }
        INDArray valArr = Nd4j.create(f, new int[]{10,1});
        return valArr;
    }

    public String readFourBytes(BufferedReader br) throws IOException {
        char[] fourBytes = new char[8];
        int i  = 0;
        while(i != 8){
            String byt = readByte(br);
            fourBytes[i] = byt.charAt(0);
            fourBytes[++i] = byt.charAt(1);
            ++i;
        }
        String s_fourBytes = new String(fourBytes);
        return s_fourBytes;
    }
    public String readByte(BufferedReader br) throws IOException {
        char[] byt = new char[2];
        int i = 0;
        while( i != 2){
            int x = br.read();
            if(x == -1){
                return "EOF";
            }
            char c = (char)x;
            if(c != ' ' && c != '\n' && c!= '\r'){
                byt[i] = c;
            }
            else{
//                x = br.read();
//                if(x == -1){
//                    return "EOF";
//                }
//                 c = (char)x;
//                byt[i] = c;
                continue;
            }
            ++i;
        }
        String s_byt = new String(byt);
        //System.out.println(s_byt);
        return s_byt;
    }
    public MNIST_Tuple[] getTraining_data() {
        return training_data;
    }

    public void setTraining_data(MNIST_Tuple[] training_data) {
        this.training_data = training_data;
    }

    public MNIST_Tuple[] getValidation_data() {
        return validation_data;
    }

    public void setValidation_data(MNIST_Tuple[] validattion_data) {
        this.validation_data = validattion_data;
    }
}
