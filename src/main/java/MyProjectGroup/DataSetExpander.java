package MyProjectGroup;

import MyProjectGroup.SGDNeuralNetwork.MNIST_Loader;
import MyProjectGroup.SGDNeuralNetwork.MNIST_Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * Created by Ian on 4/1/2016.
 * TESTING
 */
public class DataSetExpander {

    public static void main(String args[]){
        /**
         * just expand a dataset
         */
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        expandDataSet(11136);
        /**
         * Going to combine to handwritten and printed sets for differentiating
         */
//        MNIST_Loader mlPrint = null;
//        try
//        {
//            FileInputStream fileIn = new FileInputStream("ExpandedPrintSetMNIST_Loader.ser");
//            ObjectInputStream in = new ObjectInputStream(fileIn);
//            mlPrint = (MNIST_Loader) in.readObject();
//            in.close();
//            fileIn.close();
//        }catch(IOException i)
//        {
//            i.printStackTrace();
//            return;
//        }catch(ClassNotFoundException c)
//        {
//            System.out.println("MNIST_Loader class not found");
//            c.printStackTrace();
//        }
//
//        MNIST_Loader mlHW = new MNIST_Loader();
//        mlHW.load_data();
//        MNIST_Tuple[] printData = concat(mlPrint.getTraining_data(), mlPrint.getValidation_data());
//        MNIST_Tuple[] hwData = concat(mlHW.getTraining_data(), mlHW.getValidation_data());
//        for(MNIST_Tuple mt : printData){
//            double[] data = new double[] {1,0};
//            mt.setValue(Nd4j.create(data, new int[]{2,1}));
//        }
//        for(MNIST_Tuple mt : hwData){
//            double[] data = new double[] {0,1};
//            mt.setValue(Nd4j.create(data, new int[]{2,1}));
//        }
//        MNIST_Tuple[] allData = concat(printData, hwData);
//        int expandedSetLen = allData.length;
//        int trainingSize = expandedSetLen * 4 / 5;
//        int validSize = expandedSetLen - trainingSize;
//        System.out.println("ExpandedSet Size" + expandedSetLen);
//        System.out.println("trainingSize" + trainingSize);
//        System.out.println("ValidationSize" + validSize);
//        MNIST_Tuple[] trainingData = new MNIST_Tuple[trainingSize];
//        MNIST_Tuple[] validationData = new MNIST_Tuple[validSize];
//        Collections.shuffle(Arrays.asList(allData));
//        for(int i = 0; i != allData.length; ++i){
//            if(i < trainingSize){
//                trainingData[i] = allData[i];
//            }
//            else{
//                validationData[i - trainingSize] = allData[i];
//            }
//        }
//        MNIST_Loader handVPrintML = new MNIST_Loader();
//        handVPrintML.setTraining_data(trainingData);
//        handVPrintML.setValidation_data(validationData);
//        saveML(handVPrintML);
    }

    private static int num = 0;

    public static void expandDataSet(int size){
        ArrayList<MNIST_Tuple> expandedSet = new ArrayList<MNIST_Tuple>();
        MNIST_Loader ml = new MNIST_Loader(size);
        ml.load_data();
        MNIST_Tuple[] trainData = ml.getTraining_data();
        MNIST_Tuple[] validData = ml.getValidation_data();
        MNIST_Tuple[] allData = concat(trainData, validData);
        for(int i = 0; i != allData.length; ++i){
            expandedSet.add(allData[i]);
        }
        for(int i = 0; i != allData.length; ++i){
            MNIST_Tuple currentTuple = allData[i];
            INDArray currentVal = currentTuple.getValue();
            Mat currentPic = indArrayToMat(currentTuple.getPic());
            Mat[] newPics = translateIntoNewPics(currentPic);
//            Imgcodecs.imwrite("TranslatedPics\\unTranslated" + num++ +".jpg", currentPic);
            for(int x = 0; x != newPics.length; ++x){
                MNIST_Tuple mt = new MNIST_Tuple();
                mt.setValue(currentVal);
                mt.setPic(matToINDArray(newPics[x]));
                expandedSet.add(mt);
            }
        }
        //next bit for saving the setting the MNIST_Loader values and saving it
        int expandedSetLen = expandedSet.size();
        int trainingSize = expandedSetLen * 4 / 5;
        int validSize = expandedSetLen - trainingSize;
        System.out.println("ExpandedSet Size" + expandedSetLen);
        System.out.println("trainingSize" + trainingSize);
        System.out.println("ValidationSize" + validSize);
        MNIST_Tuple[] trainingData = new MNIST_Tuple[trainingSize];
        MNIST_Tuple[] validationData = new MNIST_Tuple[validSize];
        Collections.shuffle(expandedSet);
        for(int i = 0; i != expandedSetLen; ++i){
            if(i < trainingSize){
                trainingData[i] = expandedSet.get(i);
            }
            else{
                validationData[i - trainingSize] = expandedSet.get(i);
            }
        }
        ml.setTraining_data(trainingData);
        ml.setValidation_data(validationData);
        saveML(ml);
    }

    private static void saveML(MNIST_Loader ml){
        String fileName = "MyExpandedPrinted_ML.ser";
        try
        {
            FileOutputStream fileOut = new FileOutputStream(fileName);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(ml);
            out.close();
            fileOut.close();
        }catch(IOException i)
        {
            i.printStackTrace();
        }
    }

    private static Mat[] translateIntoNewPics(Mat pic){
//        Imgcodecs.imwrite("TranslatedPics\\unTranslated" + num +".jpg", pic);
        Mat transUp = transUp(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedUp" + num +".jpg", transUp);
        Mat transDown= transDown(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedDown" + num +".jpg", transDown);
        Mat transLeft = transLeft(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedLeft" + num +".jpg", transLeft);
        Mat transRight = transRight(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedRight" + num +".jpg", transRight);
        Mat transUpRight = transUpRight(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedUpRight" + num +".jpg", transUpRight);
        Mat transDownRight = transDownRight(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedDownRight" + num +".jpg", transDownRight);
        Mat transUpLeft = transUpLeft(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedUpLeft" + num +".jpg", transUpLeft);
        Mat transDownLeft = transDownLeft(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedDownLeft" + num++ +".jpg", transDownLeft);
//        Mat transRightSmall = transRightSmall(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedRightSmall" + num +".jpg", transUpRight);
//        Mat transLeftSmall = transLeftSmall(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedLeftSmall" + num +".jpg", transUpRight);
//        Mat transDownSmall = transDownSmall(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedDownSmall" + num +".jpg", transUpRight);
//        Mat transUpSmall = transUpSmall(pic);
//        Imgcodecs.imwrite("TranslatedPics\\translatedUpSmall" + num +".jpg", transUpRight);
        return new Mat[]{transUp,transDown,transLeft,transRight, transUpLeft, transUpRight, transDownLeft, transDownRight,};
    }
    private static Mat transUpRight(Mat pic){
        return transRight(transUp(pic));
    }
    private static Mat transUpLeft(Mat pic){
        return transLeft(transUp(pic));
    }
    private static Mat transDownLeft(Mat pic){
        return transLeft(transDown(pic));
    }
    private static Mat transDownRight(Mat pic){
        return transRight(transDown(pic));
    }
    private static Mat transRight(Mat pic) {
        double[][] picData = getColArrayRep(pic);
        int rightMostColofNum = 0;
        boolean finished = false;
        for(int x = pic.cols()-1; x != 0 && !finished; --x){
//            double[] col = pic.get(0,x);
            double[] col = picData[x];
            for(int y = 0; y != col.length; ++y){
                if(col[y] != 0){
                    rightMostColofNum = x;
                    finished = true;
                    break;
                }
            }
        }

        int offset = (pic.cols() - rightMostColofNum) * 3 / 4;
//        for(int x = rightMostColofNum; x != 0; --x){
////            double[] col = pic.get(0,x);
////            pic.put(0,x+offset,col);
//            picData[x+offset] = picData[x];
//        }
//        double[] buffer = pic.get(0,0);
//        pic.put(0,0, buffer);
//        return pic;
        byte[] data = new byte[pic.rows() * pic.cols() * pic.channels()];
        pic.get(0,0,data);
        Mat transPic = Mat.zeros(pic.rows(), pic.cols(), pic.type());
//        transPic.put(pic.rows() - offset,0,border);
        transPic.put(0,offset,data);
//        Imgcodecs.imwrite("TranslatedPics\\translatedRight" + num +".jpg", transPic);
        return transPic;
    }

    private static Mat transRightSmall(Mat pic){
        byte[] data = new byte[pic.rows() * pic.cols() * pic.channels()];
        pic.get(0,0,data);
        Mat transPic = Mat.zeros(pic.rows(), pic.cols(), pic.type());
//        transPic.put(pic.rows() - offset,0,border);
        transPic.put(0,1,data);
//        Imgcodecs.imwrite("TranslatedPics\\translatedRight" + num +".jpg", transPic);
        return transPic;
    }


    private static Mat transLeft(Mat pic) {
        double[][] picData = getColArrayRep(pic);
        int leftMostColofNum = 0;
        boolean finished = false;
        for(int x = 0; x != pic.cols() && !finished; ++x){
            double[] col = picData[x];
            for(int y = 0; y != col.length; ++y){
                if(col[y] != 0){
                    leftMostColofNum = x;
                    finished = true;
                    break;
                }
            }
        }

        int offset = leftMostColofNum * 3 / 4;
//        for(int x = leftMostColofNum; x != pic.cols(); ++x){
////            double[] col = pic.get(0,x);
////            pic.put(0,x-offset,col);
//            picData[x-offset] = picData[x];
//
//        }
//        double[] buffer = pic.get(0,0,buffer);
//        pic.put(0,0, buffer);
//        return pic;
        byte[] data = new byte[pic.rows() * pic.cols() * pic.channels()];
        pic.get(0,offset,data);
        Mat transPic = Mat.zeros(pic.rows(), pic.cols(), pic.type());
//        transPic.put(pic.rows() - offset,0,border);
//        transPic.put(0,-offset,data);
        transPic.put(0,0,data);
//        Imgcodecs.imwrite("TranslatedPics\\translatedLeft" + num +".jpg", transPic);
        return transPic;
    }

    private static Mat transLeftSmall(Mat pic){
        byte[] data = new byte[pic.rows() * pic.cols() * pic.channels()];
        pic.get(0,1,data);
        Mat transPic = Mat.zeros(pic.rows(), pic.cols(), pic.type());
//        transPic.put(pic.rows() - offset,0,border);
//        transPic.put(0,-offset,data);
        transPic.put(0,0,data);
//        Imgcodecs.imwrite("TranslatedPics\\translatedLeft" + num +".jpg", transPic);
        return transPic;
    }

    private static Mat transUp(Mat pic) {
        double[][] picData = getArrayRep(pic);
        int topRowOfNum = 0;
        boolean finished = false;
        for(int y =0; y != pic.rows() && !finished; ++y){
            double[] row = picData[y];
            for(int x = 0; x != row.length; ++x){
                if(row[x] != 0){
                    topRowOfNum = y;
                    finished = true;
                    break;
                }
            }
        }
        int offset = topRowOfNum * 3 / 4;
//        for(int y = topRowOfNum; y != pic.cols(); ++y){
//            double[] row = picData[y];
//            picData[y-offset] = picData[y];
////            pic.put(y-offset,0, row);
//        }
//        double[] buffer = toSingleBuffer(picData);
//        pic.put(0,0,buffer);
//        return pic;
//        double[] data = new double[pic.rows() * pic.cols() * pic.channels()];
//        byte[] border = new byte[offset * pic.cols() * pic.channels()];
//        Arrays.fill(border,(byte)0);
        byte[] data = new byte[pic.rows() * pic.cols() * pic.channels()];
        pic.get(offset,0,data);
        Mat transPic = Mat.zeros(pic.rows(), pic.cols(), pic.type());
//        transPic.put(pic.rows() - offset,0,border);
//        transPic.put(-offset,0,data);
        transPic.put(0,0,data);
//        Imgcodecs.imwrite("TranslatedPics\\translatedUp" + num +".jpg", transPic);
        return transPic;
    }

    private static Mat transUpSmall(Mat pic){
        byte[] data = new byte[pic.rows() * pic.cols() * pic.channels()];
        pic.get(1,0,data);
        Mat transPic = Mat.zeros(pic.rows(), pic.cols(), pic.type());
//        transPic.put(pic.rows() - offset,0,border);
//        transPic.put(-offset,0,data);
        transPic.put(0,0,data);
//        Imgcodecs.imwrite("TranslatedPics\\translatedUp" + num +".jpg", transPic);
        return transPic;
    }

    private static Mat transDown(Mat pic) {
        double[][] picData = getArrayRep(pic);
        int bottomRowOfNum = 0;
        boolean finished = false;
        for(int y = pic.rows() - 1; y != 0 && !finished; --y){
            double[] row = picData[y];
            for(int x = 0; x != row.length; ++x){
                if(row[x] != 0){
                    bottomRowOfNum = y;
                    finished = true;
                    break;
                }
            }
        }
        int offset = (pic.rows() - bottomRowOfNum) * 3 / 4;
//        for(int y = bottomRowOfNum; y != 0 ; --y){
////            double[] row = pic.get(y,0);
//            double[] row = picData[y];
//            picData[y + offset] = row;
////            pic.put(y+offset,0, row);
//        }
//        double [] buffer = toSingleBuffer(picData);
//        pic.put(0,0, buffer);
//        return pic;
//        double[] data = new double[pic.rows() * pic.cols() * pic.channels()];
        byte[] data = new byte[pic.rows() * pic.cols() * pic.channels()];
//        byte[] border = new byte[offset * pic.cols() * pic.channels()];
//        Arrays.fill(border,(byte)0);
        pic.get(0,0,data);
        Mat transPic = Mat.zeros(pic.rows(), pic.cols(), pic.type());
//        transPic.put(0,0,border);
        if(offset != 0){
            transPic.put(offset-1,0,data);
        }
        else{
            transPic.put(offset,0,data);
        }
//        Imgcodecs.imwrite("TranslatedPics\\translatedDown" + num +".jpg", transPic);
        return transPic;
    }

    private static Mat transDownSmall(Mat pic){
        byte[] data = new byte[pic.rows() * pic.cols() * pic.channels()];
        pic.get(0,0,data);
        Mat transPic = Mat.zeros(pic.rows(), pic.cols(), pic.type());
//        transPic.put(0,0,border);
            transPic.put(1,0,data);
//        Imgcodecs.imwrite("TranslatedPics\\translatedDown" + num +".jpg", transPic);
        return transPic;
    }

    private static INDArray matToINDArray(Mat cell){
//        String fileName1 = "InConversion\\dataCell" + num + ".jpg";
//        Imgcodecs.imwrite(fileName1, cell);
//        double[] dataCell = new double[cell.rows() * cell.cols() * cell.channels()];
//        for(int y = 0; y != cell.size().height; ++y){
//            for(int x = 0; x != cell.size().width; ++x, ++i){
//                double temp = cell.get(y,x)[0];
//                if(temp != 0){
//                    temp = temp;
//                }
//                dataCell[i] = temp;
//            }
//        }

        Mat scaled = new Mat(new Size(28,28), CvType.CV_8UC1);
        //use instead
//        Imgproc.resize(cell,scaled, scaled.size(),0,0,Imgproc.INTER_CUBIC);
        Imgproc.resize(cell,scaled, scaled.size());
//        String fileName2 = "InConversion\\cell" + num + ".jpg";
//        Imgcodecs.imwrite(fileName2, scaled);
        double[] data = new double[scaled.rows() * scaled.cols() * scaled.channels()];
        int i = 0;
        for(int y = 0; y != scaled.size().height; ++y){
            for(int x = 0; x != scaled.size().width; ++x, ++i){
                double temp = scaled.get(y,x)[0] / 255;

                if(temp != 0){
                    temp = temp;
                }
                data[i] = temp;
            }
        }
//        String text = Arrays.toString(data);
//        try {
//            PrintWriter out = new PrintWriter("6.txt");
//            out.println(text);
//            out.close();
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        }
        INDArray vals = Nd4j.create(data, new int[]{784,1});
//        num++;
//        System.out.println(verify(scaled, vals));
        return vals;
    }

    private static Mat indArrayToMat(INDArray pic){
        double[] dataFromMNIST = new double[784];
        for(int x = 0; x != 784; x++){
            dataFromMNIST[x] = pic.getDouble(x,0) * 255;
        }
        Mat mat = new Mat(new Size(28,28), CvType.CV_8UC1);
        mat.put(0,0,dataFromMNIST);
        return mat;
    }

    private static MNIST_Tuple[] concat(MNIST_Tuple[] a, MNIST_Tuple[] b) {
        int aLen = a.length;
        int bLen = b.length;
        MNIST_Tuple[] c= new MNIST_Tuple[aLen+bLen];
        System.arraycopy(a, 0, c, 0, aLen);
        System.arraycopy(b, 0, c, aLen, bLen);
        return c;
    }

    public static double[][] getArrayRep(Mat pic){
        double[][] picData = new double[pic.rows()][pic.cols()];
        for(int y = 0; y != pic.rows(); ++y){
            for(int x = 0; x != pic.cols(); ++x){
                picData[y][x] =  pic.get(y,x)[0];
            }
        }
        return picData;
    }

    private static double[][] getColArrayRep(Mat pic) {
        double[][] picData = new double[pic.cols()][pic.rows()];
        for(int x = 0; x != pic.cols(); ++x){
            for(int y = 0; y != pic.rows(); ++y){
                picData[x][y] =  pic.get(y,x)[0];
            }
        }
        return picData;
    }

    private static double[] toSingleBufferFromColRep(double[][] picData) {
        double[] buffer = new double[picData.length * picData[0].length];
        int i = 0;
        for(int x = 0; x != picData.length; ++x){
            for(int y = 0; y != picData[x].length; ++y){
                buffer[i] = picData[y][x];
            }
        }
        return buffer;
    }

    private static double[] toSingleBuffer(double[][] picData) {
    double[] buffer =  new double[picData.length * picData[0].length];
    int i = 0;
    for(int y = 0; y != picData.length; ++y){
        for(int x = 0; x != picData[y].length; ++x, ++i){
            buffer[i] = picData[y][x];
        }
    }
    return buffer;
}

    //in place blob detection on an binary image where blobs are white and background is black
    public static void simpleBlobDetect(Mat image, Scalar blobColor, Scalar backgroundColor){
        //hacky blob detect
        int count=0;
        int max=-1;
        Point maxPt = null;
        Mat imageMask = Mat.zeros(image.rows()+2, image.cols()+2, CvType.CV_8UC1);
//        System.out.println("outerbox height: " + outerBox.size().height + ", width: " + outerBox.size().width);
//        System.out.println("outerboxMask rows:" + outerboxMask.rows() + ", cols: " + outerboxMask.cols());
//        Imgcodecs.imwrite("curious.jpg", outerBox);
        for(int y=0;y<image.size().height;y++)
        {
            for(int x=0;x<image.size().width;x++)
            {
                if(image.get(y,x)[0] >= 128)
                {
//                    System.out.println("y: " + y + "x: " +x);
                    int area = Imgproc.floodFill(image, imageMask, new Point(x,y), new Scalar(64,0,0));
                    if(area > max)
                    {
                        maxPt = new Point(x,y);
                        max = area;
                    }
                }
            }
        }
        if(maxPt != null){
            imageMask = Mat.zeros(image.rows()+2, image.cols()+2, CvType.CV_8UC1);
            Imgproc.floodFill(image, imageMask, maxPt, blobColor);
            //turn everything but biggest blob to background color
            imageMask = Mat.zeros(image.rows()+2, image.cols()+2, CvType.CV_8UC1);
            for(int y=0;y<image.size().height;y++)
            {
                for(int x=0;x<image.size().width;x++)
                {
                    if(image.get(y,x)[0] != 255 && x != maxPt.x && y != maxPt.y)
                    {
                        Imgproc.floodFill(image, imageMask, new Point(x,y), backgroundColor);
                    }
                }
            }
        }
    }
}
