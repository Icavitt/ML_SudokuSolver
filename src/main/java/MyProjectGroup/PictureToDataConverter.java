package MyProjectGroup;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;

/**
 * Created by Ian on 3/29/2016.
 */
public class PictureToDataConverter {
    private File parentDirectory;
    private File labels;
    private File data;
    private StringBuilder labelString;
    private StringBuilder dataString;
    private int hexByteAppendSpaceLabel = 2;
    private int hexByteAppendSpaceData = 2;
    BufferedWriter labelWriter;
    BufferedWriter dataWriter;

    public static  void main( String[] args ){
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        PictureToDataConverter ptdc = new PictureToDataConverter();
////        ptdc.appendLabelHeader();
////        for(int i = 0; i != 100; ++i){
////            ptdc.appendHexByteLabel(i, ptdc.labelString);
////        }
////        ptdc.updateEntriesValue(100, ptdc.labelString);
////        System.out.println(ptdc.labelString.toString());
//        File parentDirectory = new File("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\");
////        File parentDirectory = new File("D:\\Downloads\\English\\Fnt");
//        ptdc.convert(parentDirectory);
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\1");
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\2");
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\3");
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\4");
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\5");
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\6");
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\7");
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\8");
////        labeler("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\9");
    }

    public static void labeler(String parentDirectory){
        File parent = new File(parentDirectory);
        int index = 0;
        for(final File fileEntry : parent.listFiles()){
            String currentName = fileEntry.getName();
            String replaceWith = parent.getName();
            currentName = parent.getAbsolutePath() + "\\" + currentName.substring(0,3) + "00" +  replaceWith + "_"+ index++ + ".jpg";
            fileEntry.renameTo(new File(currentName));
//            System.out.println("Labeled File: " + currentName);
        }

    }

    public PictureToDataConverter(String parentDirectory){
        this();
        this.parentDirectory = new File(parentDirectory);
        labels = new File(parentDirectory + "\\my-printed-labels.txt");
        data = new File(parentDirectory + "\\my-printed-data.txt");
//        convert(this.parentDirectory);
    }

    public PictureToDataConverter(){
        labelString = new StringBuilder();
        dataString = new StringBuilder();
        try {
            labelWriter = new BufferedWriter(new FileWriter(new File("my-printed-labels.txt")));
            dataWriter = new BufferedWriter(new FileWriter(new File("my-printed-data.txt")));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void convert(File parentDirectory){
        int pics = 0;
//        try {
//            labelWriter.write(appendLabelHeaderRet());
//            dataWriter.write(appendDataHeaderRet());
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        StringBuilder sb1 = new StringBuilder();
        sb1.append(appendLabelHeaderRet());
        StringBuilder sb2 = new StringBuilder();
        sb2.append(appendDataHeaderRet());
        for(final File fileEntry : parentDirectory.listFiles()){
            if(fileEntry.getName().contains(".jpg")){
                ++pics;
                Mat pic = Imgcodecs.imread(fileEntry.getName());
//                Mat pic = Imgcodecs.imread(fileEntry.getAbsolutePath() + "" + fileEntry.getName(),0);
                Mat dst = new Mat(new Size(28,28), CvType.CV_8UC1);
                Imgproc.resize(pic,dst, dst.size());
                /**
                 * uncomment below if converting images with a white background
                 */
//                Core.bitwise_not(dst, dst);
                appendLabel(fileEntry.getName(), sb1);
                appendData(dst, sb2);
                if(pics%1000 == 0) {
                    try {
                        labelWriter.write(sb1.toString());
                        dataWriter.write(sb2.toString());
                        System.out.println("write");
                        sb1 = new StringBuilder();
                        sb2 = new StringBuilder();
                        labelWriter.flush();
                        dataWriter.flush();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        System.out.println("num of pics: " + pics);
        if(pics != 11136){
            System.out.println("error");
        }
        try {
            labelWriter.write(sb1.toString());
            dataWriter.write(sb2.toString());
            labelWriter.flush();
            dataWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
//        updateEntriesValue(pics, labelString);
//        updateEntriesValue(pics, dataString);
//        String label = labelString.toString();
//        String data = dataString.toString();
    }

    private String appendLabelHeaderRet(){
        StringBuilder sb = new StringBuilder();
        appendFourHexBytes(2049, sb);
        appendFourHexBytes(0, sb);
        return sb.toString();
    }
    private String appendDataHeaderRet(){
        StringBuilder sb = new StringBuilder();
        appendFourHexBytes(2051, sb);
        appendFourHexBytes(0, sb);
        appendFourHexBytes(28, sb);
        appendFourHexBytes(28, sb);
        return sb.toString();
    }
    private void updateEntriesValue(int pics, StringBuilder dataString) {
        dataString.delete(9,19);
        dataString.insert(10, getFourByteHexValue(pics));
    }

    private String getFourByteHexValue(int num){
        StringBuilder hex = new StringBuilder();
        hex.append(Integer.toHexString(num));
        if(hex.length() == 8){
            hex.insert(4," ");
        }
        if(hex.length() < 8 ){
            if(hex.length() > 4){
                hex.insert(4," ");
                while(hex.length() < 9){
                    hex.insert(0,0);
                }
            }
            if(hex.length() < 4){
                String leading = "0000 ";
                while(hex.length() < 4){
                    hex.insert(0,0);
                }
                hex.insert(0,leading);
            }
        }
        hex.append(" ");
        return  hex.toString();
    }

    private void appendData(Mat dst, StringBuilder sb) {
        for(int y = 0; y != dst.size().height; ++y){
            for(int x = 0; x!= dst.size().width; ++x){
                double pixel = dst.get(y,x)[0];
                int value = (int)pixel;
                appendHexByteData(value, sb);
            }
        }
    }

    private void appendLabelHeader(){
        appendFourHexBytes(2049, labelString);
        appendFourHexBytes(0, labelString);
    }
    private void appendDataHeader(){
        appendFourHexBytes(2051, dataString);
        appendFourHexBytes(0, dataString);
        appendFourHexBytes(28, dataString);
        appendFourHexBytes(28, dataString);
    }

    private void appendLabel(String name, StringBuilder sb) {
        name = name.substring(3,6);
//        int val = Integer.parseInt(name) - 1;
        int val = Integer.parseInt(name);
        appendHexByteLabel(val, sb);
    }

    private void appendFourHexBytes(int val, StringBuilder str){
        StringBuilder hex = new StringBuilder();
        hex.append(Integer.toHexString(val));
        if(hex.length() == 8){
            hex.insert(4," ");
        }
        if(hex.length() < 8 ){
            if(hex.length() > 4){
                hex.insert(4," ");
                while(hex.length() < 9){
                    hex.insert(0,0);
                }
            }
            if(hex.length() < 4){
                String leading = "0000 ";
                while(hex.length() < 4){
                    hex.insert(0,0);
                }
                hex.insert(0,leading);
            }
        }
        hex.append(" ");
        str.append(hex);
    }
    private void appendHexByteLabel(int val, StringBuilder str){
        appendHexByte(val, str);
        if(hexByteAppendSpaceLabel % 2 == 1){
            str.append(" ");
        }
        hexByteAppendSpaceLabel++;
    }
    private void appendHexByteData(int val, StringBuilder str){
        appendHexByte(val, str);
        if(hexByteAppendSpaceData % 2 == 1){
            str.append(" ");
        }
        hexByteAppendSpaceData++;
    }
    private void appendHexByte(int val, StringBuilder str) {
        String hex = Integer.toHexString(0xFF & (byte)val);
        if(hex.length() == 1){
            hex = 0 + hex;
        }
//        System.out.println(hex);
        str.append(hex);
    }

//    private String toHexString(byte[] bytes) {
//        StringBuilder hexString = new StringBuilder();
//
//        for (int i = 0; i < bytes.length; i++) {
//            String hex = Integer.toHexString(0xFF & bytes[i]);
//            if (hex.length() == 1) {
//                hexString.append('0');
//            }
//            hexString.append(hex);
//        }
//        return hexString.toString();
//    }

    public File getParentDirectory() {
        return parentDirectory;
    }

    public void setParentDirectory(File parentDirectory) {
        this.parentDirectory = parentDirectory;
    }
}
