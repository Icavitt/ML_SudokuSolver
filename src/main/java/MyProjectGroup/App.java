package MyProjectGroup;

//Packages for ND4J...may need to get BLAS
import MyProjectGroup.SGDNeuralNetwork.*;
import com.github.sarxos.webcam.WebcamPanel;
import com.github.sarxos.webcam.WebcamResolution;
import com.sun.rowset.WebRowSetImpl;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
//Packages for Webcam Capture and related file IO
import com.github.sarxos.webcam.Webcam;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.utils.Converters;

import java.awt.*;
//import java.awt.List;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.*;
import java.util.List;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.util.*;



/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
//        Webcam webcam = Webcam.getDefault();
//        if (webcam != null) {
//            System.out.println("Webcam: " + webcam.getName());
//        } else {
//            System.out.println("No webcam detected");
//        }
/**
 * Start image processing code/tests
 */
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
/**
 * Below is used to create an example of a handwritten digit from MNIST sample
 */
//        MNIST_Loader ml = new MNIST_Loader(10160);
//        ml.load_data();
//        MNIST_Tuple[] mts = ml.getTraining_data();
//        int i = 0;
//        for(MNIST_Tuple mt : mts){
//            INDArray ia = mt.getValue();
//            double val = 0;
//            int valj = 0;
//            for(int j = 0; j != 10; ++j){
//                double tempD = ia.getDouble(j,0);
//                if(tempD > val){
//                    val = tempD;
//                    valj = j;
//                }
//            }
//            if(valj == 6){
////                System.out.println(mt.getPic());
//                double[] dataFromMNIST = new double[784];
//                for(int x = 0; x != 784; x++){
//                    dataFromMNIST[x] = mt.getPic().getDouble(x,0) * 255;
//                }
//                Mat mnist6 = new Mat(new Size(28,28), CvType.CV_8UC1);
//                mnist6.put(0,0,dataFromMNIST);
//                Imgcodecs.imwrite("ExampleMnistCharacters\\MNIST6" + i++ +".jpg", mnist6);
////                String text = Arrays.toString(dataFromMNIST);
////                try {
////                    PrintWriter out = new PrintWriter("Mnist6.txt");
////                    out.println(text);
////                    out.close();
////                } catch (FileNotFoundException e) {
////                    e.printStackTrace();
////                }
//            }
//        }
/**
 * Image processing / main code  here
 */
        NeuralNet nn = loadNNFromFile("98.91743576951858% My Print NN.ser");
//        File parentDirectory = new File("D:\\OneDrive\\School\\SeniorProj\\Projects\\ND4JTest\\");
//        for(final File fileEntry : parentDirectory.listFiles()){
//            if(fileEntry.getName().contains(".jpg")){
//                topRowOfPuzzle = 0;
//                bottomRowofPuzzle = 0;
//                Mat sudoku = Imgcodecs.imread(fileEntry.getName(),0);
                final Webcam webcam = Webcam.getDefault();
                JFrame window = new JFrame("Test webcam panel");
                JTextArea jOut = new JTextArea(10,10);
                jOut.setEditable(false);
                JTextField jIn = new JTextField(10);
                TextFieldStreamer ts = new TextFieldStreamer(jIn);
                jIn.addKeyListener(ts);
                System.setIn(ts);
                JPanel ioPanel = new JPanel(new GridLayout(2,1));
                ioPanel.add(new JScrollPane(jOut));
                ioPanel.add(jIn);
                window.add(ioPanel, BorderLayout.PAGE_END);
                CustomOutputStream outputStream = new CustomOutputStream(jOut);
                PrintStream printStream = new PrintStream(outputStream);
                System.setOut(printStream);
                window.setResizable(true);
                window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                window.pack();
                window.setVisible(true);
                if(webcam != null){
                    System.out.println("Webcam: " + webcam.getName());
                    webcam.setViewSize(WebcamResolution.VGA.getSize());
                    WebcamPanel panel = new WebcamPanel(webcam);
                    Button takePic = new Button("Take Picture");
                    takePic.addActionListener(new ActionListener() {
                        public void actionPerformed(ActionEvent e) {
                            BufferedImage image = webcam.getImage();
                            try {
                                ImageIO.write(image, "PNG", new File("test.png"));
                            } catch (IOException e1) {
                                e1.printStackTrace();
                            }
                        }
                    });
                    window.add(takePic, BorderLayout.EAST);
                    window.add(panel);
                }
                else{
                    System.out.print("No webcam detected");
                }
                Mat sudoku = Imgcodecs.imread("sudoku.jpg",0);
                if(sudoku.size().width > 400){
                    sudoku = scale(sudoku);
                }
//                System.out.println("name: " +fileEntry.getName()+"\n size: " + sudoku.size());
                Mat outerBox = new Mat(sudoku.size(), CvType.CV_8UC1);
                //supposed to make extracting gridlines easier
                Imgproc.GaussianBlur(sudoku, sudoku, new Size(11,11), 0);
                //thresholding the image
                Imgproc.adaptiveThreshold(sudoku, outerBox, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 5, 2);
                //inverting thresholded colors
                Core.bitwise_not(outerBox, outerBox);
        //        Imgcodecs.imwrite("postThreshInv.jpg", outerBox);
                byte[] data = {0,1,0,1,1,1,0,1};
                Mat kernel = new Mat(3,3, CvType.CV_8U);
                kernel.put(0,0,data);
                //"fills in cracks" craeted from thresholding
                Imgproc.dilate(outerBox, outerBox, kernel);
                simpleBlobDetect(outerBox, new Scalar(255,255,255), new Scalar(0,0,0));
                Imgproc.erode(outerBox, outerBox, kernel);
                //calculate lines for outer grid
                Mat lines = new Mat();
//                Imgproc.HoughLines(outerBox,lines,1, Math.PI/180,200);
                Imgproc.HoughLinesP(outerBox,lines,1, Math.PI/180,200);
                for(int i = 0; i != lines.rows(); ++i){
                    double[] line = lines.get(i,0);
                    Scalar bgr = new Scalar(128,0,0);
                    drawLineProb(line, outerBox, bgr);
                }
        //        Imgcodecs.imwrite("houghLines.jpg", outerBox);
                //merge closely related lines
//                mergeLines(lines, outerBox);
                Imgcodecs.imwrite("lines.png", outerBox);
                //        for(int i = 0; i != lines.rows(); ++i){
        //            double[] line = lines.get(i,0);
        //            double p = line[0];
        //            double theta = line[1];
        //            if(p ==0 && theta == -100){
        //                continue;
        //            }
        //            if(line[0] != 0){
        //                double m = -1/Math.tan(line[1]);
        //                double c = line[0]/Math.sin(line[1]);
        //                Imgproc.line(outerBox, new Point(0,c), new Point(outerBox.size().width, m*outerBox.size().width+c), new Scalar(128,0,0));
        //            }
        //            else{
        //                Imgproc.line(outerBox, new Point(line[0], 0), new Point(line[0], outerBox.size().height), new Scalar(128,0,0));
        //            }
        //
        //        }
        //        Imgcodecs.imwrite("afterMerge.jpg", outerBox);
                /**
                 * FIND OUTER MOST LINES AND POINTS OLD WAY
                 */
//                double[] topEdge = {10000, 10000};
//                double topYIntercept = 100000, topXIntercept = 0;
//                double[] bottomEdge = {-10000, -10000};
//                double bottomYIntercept = 0, bottomXIntercept = 0;
//                double[] leftEdge = {10000, 10000};
//                double leftXIntercept = 100000, leftYIntercept = 0;
//                double[] rightEdge = {-10000, -10000};
//                double rightXIntercept = 0, rightYIntercept = 0;
//                for(int i=0; i<lines.rows(); ++i){
//                    double[] line = lines.get(i,0);
//                    double p = line[0];
//                    double theta = line[1];
//                    if(p ==0 && theta == -100){
//                        continue;
//                    }
//                    double xIntercept, yIntercept;
//                    xIntercept = p/Math.cos(theta);
//                    yIntercept = p/(Math.cos(theta)*Math.sin(theta));
//                    if(theta > Math.PI * 80/180 && theta < Math.PI * 100/180){
//                        if(p < topEdge[0]){
//                            topEdge = line;
//                        }
//                        //else if...?
//                        if(p > bottomEdge[0]){
//                            bottomEdge = line;
//                        }
//                    }
//                    else if(theta < Math.PI * 10/180 || theta > Math.PI * 170/180){
//                        if(xIntercept > rightXIntercept){
//                            rightEdge = line;
//                            rightXIntercept = xIntercept;
//                        }
//                        else if( xIntercept <= leftXIntercept){
//                            leftEdge = line;
//                            leftXIntercept = xIntercept;
//                        }
//                    }
//                }
        //        drawLine(topEdge, sudoku, new Scalar(0,0,255));
        //        drawLine(bottomEdge, sudoku, new Scalar(0,0,255));
        //        drawLine(leftEdge, sudoku, new Scalar(0,0,255));
        //        drawLine(rightEdge, sudoku, new Scalar(0,0,255));
        //        Imgcodecs.imwrite("edges.jpg", sudoku);
                //now need to calculate intersect of lines
                //first find 2 points on each
                double[] topLine = topMostLine(lines);
                double[] bottomLine = bottomMostLine(lines);
                double[] rightLine = rightMostLine(lines);
                double[] leftLine = leftMostLine(lines);
                drawLineProb(topLine,sudoku,new Scalar(0,0,255));
                drawLineProb(rightLine,sudoku,new Scalar(0,0,255));
                Imgcodecs.imwrite("plz.jpg", sudoku);
                Point topLeft = computeIntersect(topLine, leftLine);
                Point topRight = computeIntersect(topLine, rightLine);
                Point bottomLeft = computeIntersect(bottomLine, leftLine);
                Point bottomRight = computeIntersect(bottomLine, rightLine);
                Point[] newPoints = new Point[] {topLeft,topRight,bottomLeft,bottomRight};
                /**
                 * old bad point finding
                 */
//                Point left1 = new Point(), left2 = new Point(), right1 = new Point(), right2 = new Point(),
//                        bottom1 = new Point(), bottom2 = new Point(), top1 = new Point(), top2 = new Point();
//                double height = outerBox.size().height;
//                double width = outerBox.size().width;
//                if(leftEdge[1] != 0){
//                    left1.x = 0;
//                    left1.y = leftEdge[0]/Math.sin(leftEdge[1]);
//                    left2.x = width;
//                    left2.y = left2.x/Math.tan(leftEdge[1]) + left1.y;
//                }
//                else{
//                    left1.y = 0;
//                    left1.x = leftEdge[0]/Math.cos(leftEdge[1]);
//                    left2.y = height;
//                    left2.x = left1.x - height * Math.tan(leftEdge[1]);
//                }
//                if(rightEdge[1] != 0){
//                    right1.x = 0;
//                    right1.y = rightEdge[0]/Math.sin(rightEdge[1]);
//                    right2.x = width;
//                    right2.y = right2.x/Math.tan(rightEdge[1]) + right1.y;
//                }
//                else{
//                    right1.y = 0;
//                    right1.x = rightEdge[0]/Math.cos(rightEdge[1]);
//                    right2.y = height;
//                    right2.x = right1.x - height * Math.tan(rightEdge[1]);
//                }
//                bottom1.x = 0;
//                bottom1.y = bottomEdge[0]/Math.sin(bottomEdge[1]);
//                bottom2.x = width;
//                bottom2.y = bottom2.x/Math.tan(bottomEdge[1]) + bottom1.y;
//                top1.x = 0;
//                top1.y = topEdge[0]/Math.sin(topEdge[1]);
//                top2.x = width;
//                top2.y = top2.x/Math.tan(topEdge[1]) + top1.y;
//                // Next, we find the intersection of  these four lines
//                double leftA = left2.y-left1.y;
//                double leftB = left1.x-left2.x;
//                double leftC = leftA*left1.x + leftB*left1.y;
//                double rightA = right2.y-right1.y;
//                double rightB = right1.x-right2.x;
//                double rightC = rightA*right1.x + rightB*right1.y;
//                double topA = top2.y-top1.y;
//                double topB = top1.x-top2.x;
//                double topC = topA*top1.x + topB*top1.y;
//                double bottomA = bottom2.y-bottom1.y;
//                double bottomB = bottom1.x-bottom2.x;
//                double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;
//                // Intersection of left and top
//                double detTopLeft = leftA*topB - leftB*topA;
//                Point ptTopLeft = new Point((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);
//
//                // Intersection of top and right
//                double detTopRight = rightA*topB - rightB*topA;
//                Point ptTopRight = new Point((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);
//                // Intersection of right and bottom
//                double detBottomRight = rightA*bottomB - rightB*bottomA;
//                Point ptBottomRight = new Point((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);// Intersection of bottom and left
//                double detBottomLeft = leftA*bottomB-leftB*bottomA;
//                Point ptBottomLeft = new Point((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);
//                double maxLength = (ptBottomLeft.x-ptBottomRight.x)*(ptBottomLeft.x-ptBottomRight.x) + (ptBottomLeft.y-ptBottomRight.y)*(ptBottomLeft.y-ptBottomRight.y);
//                double temp = (ptTopRight.x-ptBottomRight.x)*(ptTopRight.x-ptBottomRight.x) + (ptTopRight.y-ptBottomRight.y)*(ptTopRight.y-ptBottomRight.y);
//
//                if(temp>maxLength) maxLength = temp;
//
//                temp = (ptTopRight.x-ptTopLeft.x)*(ptTopRight.x-ptTopLeft.x) + (ptTopRight.y-ptTopLeft.y)*(ptTopRight.y-ptTopLeft.y);
//
//                if(temp>maxLength) maxLength = temp;
//
//                temp = (ptBottomLeft.x-ptTopLeft.x)*(ptBottomLeft.x-ptTopLeft.x) + (ptBottomLeft.y-ptTopLeft.y)*(ptBottomLeft.y-ptTopLeft.y);
//
//                if(temp>maxLength) maxLength = temp;
//                maxLength = Math.sqrt(maxLength);
                ArrayList<Point> srcData = new ArrayList<Point>();
                srcData.add(topLeft);
                srcData.add(topRight);
                srcData.add(bottomRight);
                srcData.add(bottomLeft);
                double maxLength = getMaxLength(srcData);
                ArrayList<Point> dstData = new ArrayList<Point>();
                dstData.add(new Point(0,0));
                dstData.add(new Point(maxLength-1, 0));
                dstData.add(new Point(maxLength-1, maxLength-1));
                dstData.add(new Point(0, maxLength-1));
//
//                Mat sudoku1 = Imgcodecs.imread(fileEntry.getName());
//                if(sudoku1.size().width > 500){
//                    sudoku1 = scale(sudoku);
//                }
//                for(Point p : newPoints){
//                    if(p.x < 0){
//                        p.x = -p.x;
//                    }
//                    if(p.y < 0){
//                        p.y = -p.y;
//                    }
//                    System.out.println("x: " + p.x + " y: " + p.y);
//                    Imgproc.circle(sudoku1, p, 20, new Scalar(0,0,255));
//                }
//                Imgcodecs.imwrite("points.jpg", sudoku1);
                Mat src = Converters.vector_Point2f_to_Mat(srcData);
                Mat dst = Converters.vector_Point2f_to_Mat(dstData);
                Mat undistorted = new Mat(new Size(maxLength, maxLength), CvType.CV_8UC1);
//                sudoku = Imgcodecs.imread(fileEntry.getName());
                sudoku = Imgcodecs.imread("sudoku.jpg",0);
                if(sudoku.size().width > 400){
                    sudoku = scale(sudoku);
                }
                Imgproc.warpPerspective(sudoku, undistorted, Imgproc.getPerspectiveTransform(src, dst), new Size(maxLength, maxLength));
                Imgcodecs.imwrite("cropped.jpg", undistorted);
                int dist = (int)Math.ceil(maxLength / 9);
                for(int j = 0; j != 9; ++j){
                    for(int i = 0; i != 9; ++i){
                        Mat currentCell = new Mat(dist, dist, CvType.CV_8UC1);
                        for(int y = 0; y != dist && j*dist+y<undistorted.cols();y++){
                            double[] cellData = new double[dist];
                            for(int x = 0; x != dist && i * dist + x < undistorted.rows(); ++x){
                                cellData[x] = undistorted.get(j*dist + y,i * dist + x)[0];
                            }
                            currentCell.put(y,0, cellData);
                        }
                        Mat blobbedCell = currentCell.clone();
                        Imgproc.adaptiveThreshold(currentCell,blobbedCell , 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 5, 2);
                        Core.bitwise_not(blobbedCell, blobbedCell);
                        String fileName;
                        blobbedCell = simpleCenteredBlobDetect(blobbedCell,new Scalar(255,255,255), new Scalar(0,0,0));
                        blobbedCell = simpleCenteredBlobDetect(blobbedCell,new Scalar(255,255,255), new Scalar(0,0,0));
//                        fileName = "ThreshCenterBlobCells\\"+j+","+i+".jpg";
//                        Imgcodecs.imwrite(fileName, blobbedCell);
//                        System.out.println(fileName);
                        boolean blank = isBlankCell(blobbedCell);
                        if(!blank){
                            INDArray cell = matToINDArray(blobbedCell);
                            int cellValue = nn.detectInteger(cell);
                            fileName = "RealTest\\cellVal: "+ cellValue + "row: "+ j + "col: "+ i +".jpg";
                            Imgcodecs.imwrite(fileName, blobbedCell);
        //                    if(nn.isPrinted(cell)){
        //                        System.out.println(j+", "+i+", CellValue: is Printed");
        //                    }
        //                    else{
        //                        System.out.println(j+", "+i+", CellValue: is Handwritten");
        //                    }
                            System.out.println(j+", "+i+", CellValue: " + cellValue);
                        }
                    }
                }
//            }
//        }

/**
 * actually run the NN
 * can come back to this for more hyper-parameter editing/other techniques to make more accurate
 * also retrain with a dataset that includes pinted digits...make new network to accomdate differetiating between printed and handwritten
 */
////        MNIST_Loader ml = new MNIST_Loader(10160);
////        ml.load_data();
//        MNIST_Loader ml = null;
//        try
//        {
//            FileInputStream fileIn = new FileInputStream("MyExpandedPrinted_ML.ser");
//            ObjectInputStream in = new ObjectInputStream(fileIn);
//            ml = (MNIST_Loader) in.readObject();
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
//        System.out.println(ml.getTraining_data().length);
//        System.out.println("Data Loaded");
//        int[] sizes = {784,100,10};
//        NeuralNet nn = new NeuralNet(sizes);
////        NeuralNet nn = null;
////        try
////        {
////            FileInputStream fileIn = new FileInputStream("90.0% NN.ser");
////            ObjectInputStream in = new ObjectInputStream(fileIn);
////            nn = (NeuralNet) in.readObject();
////            in.close();
////            fileIn.close();
////        }catch(IOException i)
////        {
////            i.printStackTrace();
////            return;
////        }catch(ClassNotFoundException c)
////        {
////            System.out.println("Employee class not found");
////            c.printStackTrace();
////        }
//        System.out.println("nn created");
//        SGDTrainer sgd = new SGDTrainer(nn);
//        sgd.setTraining_data(ml.getTraining_data());
//        sgd.setValidation_data(ml.getValidation_data());
//        System.out.println("sgd data set");
//        sgd.train(10,20,.1,.0001,true);
/**
 * From here below is hand v print NN stuff
 */
//        MNIST_Loader ml = null;
//        try
//        {
//            FileInputStream fileIn = new FileInputStream("HandVPrintML_Loader.ser");
//            ObjectInputStream in = new ObjectInputStream(fileIn);
//            ml = (MNIST_Loader) in.readObject();
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
//        System.out.println(ml.getTraining_data().length);
//        System.out.println("Data Loaded");
//        int[] sizes = {784,150,2};
//        NeuralNet nn = new NeuralNet(sizes);
//        System.out.println("nn created");
//        SGDTrainer sgd = new SGDTrainer(nn);
//        sgd.setTraining_data(ml.getTraining_data());
//        sgd.setValidation_data(ml.getValidation_data());
//        System.out.println("sgd data set");
//        sgd.train(10,30,.1,.0001,true);
/**
 * Sudoku solving
 */
//        int[][] board = new int[9][9];
//        board[0][3] = 6;    board[0][5] = 4;    board[0][6] = 7;
//        board[1][0] = 7;    board[1][2] = 6;    board[1][8] = 9;
//        board[2][5] = 5;    board[2][7] = 8;
//        board[3][1] = 7;    board[3][4] = 2;    board[3][7] = 9;    board[3][8] = 3;
//        board[4][0] = 8;    board[4][8] = 5;
//        board[5][0] = 4;    board[5][1] = 3;    board[5][4] = 1;    board[5][7] = 7;
//        board[6][1] = 5;    board[6][3] = 2;
//        board[7][0] = 3;    board[7][6] = 2;    board[7][8] = 8;
//        board[8][2] = 2;    board[8][3] = 3;    board[8][5] = 1;
//        SimpleSudokuSolver.solve(board);
    }

    private static double getMaxLength(ArrayList<Point> srcData) {
        Point topLeft = srcData.get(0);     Point topRight = srcData.get(1);
        Point bottomLeft = srcData.get(3);  Point bottomRight = srcData.get(2);
        double maxLength = (bottomLeft.x-bottomRight.x)*(bottomLeft.x-bottomRight.x) + (bottomLeft.y-bottomRight.y)*(bottomLeft.y-bottomRight.y);
        double temp = (topLeft.x-topRight.x)*(topLeft.x-topRight.x) + (topLeft.y-topRight.y)*(topLeft.y-topRight.y);
        if(temp > maxLength)
            maxLength = temp;
        temp = (topLeft.x-bottomLeft.x)*(topLeft.x-bottomLeft.x) + (topLeft.y-bottomLeft.y)*(topLeft.y-bottomLeft.y);
        if(temp > maxLength)
            maxLength = temp;
        temp = (topRight.x-bottomRight.x)*(topRight.x-bottomRight.x) + (topRight.y-bottomRight.y)*(topRight.y-bottomRight.y);
        if(temp > maxLength)
            maxLength = temp;
        return Math.sqrt(maxLength);
    }

    private static Point computeIntersect(double[] bottomLine, double[] rightLine) {
        double x1 = bottomLine[0], y1 = bottomLine[1], x2 = bottomLine[2], y2 = bottomLine[3];
        double x3 = rightLine[0], y3 = rightLine[1], x4 = rightLine[2], y4 = rightLine[3];
        Point pt = new Point();     pt.x = 0.0;     pt.y = 0.0;
        double d = ((x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4));
        pt.x = ( ((x1*y2 - y1*x2) * (x3-x4)) - ((x1-x2) * (x3*y4 - y3*x4)) );
        pt.x = pt.x / d;
        pt.y = ( ((x1*y2 - y1*x2) * (y3-y4)) - ((y1-y2) * (x3*y4 - y3*x4)) );
        pt.y = pt.y / d;
        return pt;
    }

    private static double[] bottomMostLine(Mat lines) {
        double maxY = 0;
        double[] topLine = new double[4];
        for(int i = 0; i != lines.rows(); ++i){
            double[] line = lines.get(i,0);
//            if(line[0] == line[2] && line[1] == line[3]){
//                //this means the line is defined by a single point and we should skip it
//                continue;
//            }
            double dist = Math.sqrt((line[0] - line[2]) * (line[0] - line[2]) + (line[3] - line[1]) * (line[3] - line[1]));
            if(dist <= 12){
                //line is very small may be a single point or have a sharp angle
                continue;
            }
            double currentY = (line[1] + line[3]) / 2;
            if(currentY > maxY){
                maxY = currentY;
                topLine = line;
            }
        }
        return topLine;
    }

    private static double[] topMostLine(Mat lines) {
        double minY = 10000000;
        double[] bottomLine = null;
        for(int i = 0; i != lines.rows(); ++i){
            double[] line = lines.get(i,0);
//            if(line[0] == line[2] && line[1] == line[3]){
//                //this means the line is defined by a single point and we should skip it
//                continue;
//            }
            double dist = Math.sqrt((line[0] - line[2]) * (line[0] - line[2]) + (line[3] - line[1]) * (line[3] - line[1]));
            if(dist <= 12){
                //line is very small may be a single point or have a sharp angle
                continue;
            }
            double currentY = (line[1] + line[3]) / 2;
            if(currentY < minY){
                minY = currentY;
                bottomLine = line;
            }
        }
        return bottomLine;
    }

    private static double[] rightMostLine(Mat lines) {
        double maxX = 0;
        double[] rightLine = null;
        for(int i = 0; i != lines.rows(); ++i){
            double[] line = lines.get(i,0);
//            if(line[0] == line[2] && line[1] == line[3]){
//                //this means the line is defined by a single point and we should skip it
//                continue;
//            }
            double dist = Math.sqrt((line[0] - line[2]) * (line[0] - line[2]) + (line[3] - line[1]) * (line[3] - line[1]));
            if(dist <= 12){
                //line is very small may be a single point or have a sharp angle
                continue;
            }
            double currentX = (line[0] + line[2]) / 2;
            if(currentX > maxX){
                maxX = currentX;
                rightLine = line;
            }
        }
        return rightLine;
    }

    private static double[] leftMostLine(Mat lines) {
        double minX = 10000000;
        double[] leftLine = null;
        for(int i = 0; i != lines.rows(); ++i){
            double[] line = lines.get(i,0);
//            if(line[0] == line[2] && line[1] == line[3]){
//                //this means the line is defined by a single point and we should skip it
//                continue;
//            }
            double dist = Math.sqrt((line[0] - line[2]) * (line[0] - line[2]) + (line[3] - line[1]) * (line[3] - line[1]));
            if(dist <= 12){
                //line is very small may be a single point or have a sharp angle
                continue;
            }
            double currentX = (line[0] + line[2]) / 2;
            if(currentX < minX){
                minX = currentX;
                leftLine = line;
            }
        }
        return leftLine;
    }

    private static void drawLineProb(double[] line, Mat outerBox, Scalar bgr) {
        Imgproc.line(outerBox, new Point(line[0],line[1]), new Point(line[2], line[3]), bgr);
    }

    private static Mat scale(Mat sudoku) {
        int scale = (int)sudoku.size().width / 400;
        Size size = new Size(sudoku.size().width / scale, sudoku.size().height / scale);
        Imgproc.resize(sudoku,sudoku, size);
        return sudoku;
    }

    static int bottomRowofPuzzle = 0;
    private static Mat cropBottom(Mat outerBox){
        double[][] picData = DataSetExpander.getArrayRep(outerBox);
//        int x1 = 0;     int x2 = (int)outerBox.size().width;
        int y1 = 0;     int y2 = (int)outerBox.size().height;
        boolean finished = false;
        if(bottomRowofPuzzle == 0){
            for(int y = outerBox.rows() - 1; y != 0 && !finished; --y){
                double[] row = picData[y];
                for(int x  = 0; x != row.length; ++x){
                    if(row[x] != 0){
                        y1 = y;
                        bottomRowofPuzzle = y1;
                        finished = true;
                        break;
                    }
                }
            }
        }
        else{
            y1 = bottomRowofPuzzle;
        }
//        double emptySpace = ((double) y1) / outerBox.height();
//        if( emptySpace >= .2){
//            int offset = y1/2;
        int offset = outerBox.rows() - y1;
        byte[] data = new byte[(outerBox.rows() - offset) * outerBox.cols() * outerBox.channels()];
        outerBox.get(0,0,data);
        outerBox = Mat.zeros(outerBox.rows() - offset, outerBox.cols(), outerBox.type());
        outerBox.put(0,0,data);
        return outerBox;
//        }
//        return outerBox;
    }
    static int topRowOfPuzzle = 0;
    private static Mat cropTop(Mat outerBox) {
        double[][] picData = DataSetExpander.getArrayRep(outerBox);
//        int x1 = 0;     int x2 = (int)outerBox.size().width;
        int y1 = 0;     int y2 = (int)outerBox.size().height;
        boolean finished = false;
        if(topRowOfPuzzle == 0){
            for(int y = 0; y != outerBox.size().width && !finished; ++y){
                double[] row = picData[y];
                for(int x  = 0; x != row.length; ++x){
                    if(row[x] != 0){
                        y1 = y;
                        topRowOfPuzzle = y1;
                        finished = true;
                        break;
                    }
                }
            }
        }
        else{
            y1 = topRowOfPuzzle;
        }
//        double emptySpace = ((double) y1) / outerBox.height();
//        if( emptySpace >= .2){
//            int offset = y1/2;
            int offset = y1;
            byte[] data = new byte[(outerBox.rows() - offset) * outerBox.cols() * outerBox.channels()];
            outerBox.get(offset,0,data);
            outerBox = Mat.zeros(outerBox.rows() - offset, outerBox.cols(), outerBox.type());
            outerBox.put(0,0,data);
            return outerBox;
//        }
//        return outerBox;
    }

    //blobbed cell should be a thresholded image where the background is black and foreground white
    private static boolean isBlankCell(Mat blobbedCell) {

//        int size = blobbedCell.rows();
//        int offset = size / 4;
//        size = size - offset;
//        int filled = 0;
//        //channels should be 1
////        byte[] buffer = new byte[size * size * blobbedCell.channels()];
////        blobbedCell.get(offset/2,offset/2,buffer);
////        int filled = 0;
//////        int unfilled = 0;
////        for(byte d : buffer){
////            if(d == 255){
////                filled++;
////            }
////            else{
//////                unfilled++;
////            }
////        }
//        for(int y = offset / 2; y != blobbedCell.size().height - offset/2; ++y){
//            for(int x = offset / 2; x != blobbedCell.size().width - offset/2; ++x){
//                double val = blobbedCell.get(y,x)[0];
//                if(val != 0){
//                    filled++;
//                }
//            }
//        }
////        Imgproc.rectangle(blobbedCell, new Point(offset/2,offset/2), new Point(offset/2+size, offset/2+size),new Scalar(64,0,0));
//        int frac = size * size / 15;
//        if(filled >= frac){
//            return false;
//        }
//        return true;
        Moments m = Imgproc.moments(blobbedCell, true);
//        System.out.println("m00: " + m.get_m00());
//        System.out.println("total area: " + blobbedCell.rows() * blobbedCell.cols());
//        System.out.println("small area: " + blobbedCell.rows()*blobbedCell.cols() / 5);
        if((int)m.get_m00() > (blobbedCell.rows()*blobbedCell.cols() / 35)){
            return false;
        }
        return true;
    }

    static int num = 10031;
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

    private static boolean verify(Mat cell, INDArray pic){
        int picIndex = 0;
        for(int y = 0; y != cell.size().height; ++y){
            for(int x = 0; x != cell.size().width; ++x){
                double cellValue = cell.get(y,x)[0] / 255;
                double picValue = pic.getDouble(picIndex,0);
                if(Math.abs(cellValue - picValue) >= .00001){
                    return false;
                }
                ++picIndex;
            }
        }
        return true;
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

    public static Mat simpleCenteredBlobDetect(Mat image, Scalar blobColor, Scalar backgroundColor){
        //hacky blob detect
        int count=0;
        int max=-1;
        Point maxPt = null;
        int size = image.rows() / 2;
        int offset = image.rows() / 30 * 12;

        Mat imageMask = Mat.zeros(image.rows()+2, image.cols()+2, CvType.CV_8UC1);
//        System.out.println("outerbox height: " + outerBox.size().height + ", width: " + outerBox.size().width);
//        System.out.println("outerboxMask rows:" + outerboxMask.rows() + ", cols: " + outerboxMask.cols());
//        Imgcodecs.imwrite("curious.jpg", outerBox);
        for(int y=offset;y<image.size().height-offset;y++)
        {
            for(int x=offset;x<image.size().width-offset;x++)
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
            Imgproc.floodFill(image, imageMask, maxPt, new Scalar(250,250,250));
            //turn everything but biggest blob to background color
            imageMask = Mat.zeros(image.rows()+2, image.cols()+2, CvType.CV_8UC1);
            for(int y=0;y<image.size().height;y++)
            {
                for(int x=0;x<image.size().width;x++)
                {
                    if(image.get(y,x)[0] != 250 && x != maxPt.x && y != maxPt.y)
                    {
                        Imgproc.floodFill(image, imageMask, new Point(x,y), backgroundColor);
                    }
                }
            }
            Imgproc.floodFill(image, imageMask, maxPt, new Scalar(255,255,255));
            return image;
        }
       else{
            image = Mat.zeros(image.rows(), image.cols(),image.type());
            return image;
        }
    }


    public static void drawLine(double[] line, Mat img, Scalar rgb){
        if(line[0] != 0){
            double m = -1/Math.tan(line[1]);
            double c = line[0]/Math.sin(line[1]);
            Imgproc.line(img, new Point(0,c), new Point(img.size().width, m*img.size().width+c), rgb);
        }
        else{
            Imgproc.line(img, new Point(line[0], 0), new Point(line[0], img.size().height), rgb);
        }
    }

    public static void mergeLines(Mat lines, Mat img){
        for(int i = 0; i != lines.rows(); ++i){
            double[] line = lines.get(i,0);
            if (line[0] == 0 && line[1] !=-1) {
                continue;
            }
            double p1 = line[0];
            double theta1 = line[1];
            Point pt1 = new Point();
            Point pt2 = new Point();
            if(theta1 > Math.PI * 45.0/180.0 && theta1 < Math.PI * 135.0/180.0){
                pt1.x = 0;
                pt1.y = p1/Math.sin(theta1);
                pt2.x = img.size().width;
                pt2.y = -pt2.x/Math.tan(theta1) + p1/Math.sin(theta1);
            }
            else{
                pt1.y = 0;
                pt1.x = p1/Math.cos(theta1);
                pt2.y = img.size().height;
                pt2.x = -pt2.y/Math.tan(theta1) + p1/Math.cos(theta1);
            }
            for(int j = 0; j != lines.rows(); ++j){
                if(i == j){
                    continue;
                }
                double[] line2 = lines.get(j,0);
                if(Math.abs(line2[0]-line[0]) < 20 && Math.abs(line2[1] - line[1]) < Math.PI * 10/180){
                    double p = line2[0];
                    double theta = line[1];

                    Point l2pt1 = new Point();
                    Point l2pt2 = new Point();
                    if(line2[1] > Math.PI * 45/180 && line2[1] < Math.PI * 135/180){
                        l2pt1.x = 0;
                        l2pt1.y = p/Math.sin(theta);
                        l2pt2.x = img.size().width;
                        l2pt2.y = -l2pt2.x/Math.tan(theta) + p/Math.sin(theta);
                    }
                    else{
                        l2pt1.y = 0;
                        l2pt1.x = p/Math.cos(theta);
                        l2pt2.y = img.size().height;
                        l2pt2.x = -pt2.y/Math.tan(theta) + p/Math.sin(theta);
                    }
                    if((l2pt1.x - pt1.x)*(l2pt1.x - pt1.x) + (l2pt1.y - pt1.y) * (l2pt1.y - pt1.y) < 64*64 &&
                      (l2pt2.x - pt2.x)*(l2pt2.x - pt2.x) + (l2pt2.y - pt2.y) * (l2pt2.y - pt2.y) < 64*64)
                    {
                        line[0] = (line[0] + line2[0])/2;
                        line[1] = (line[1] + line2[1])/2;

                        line2[0] = 0;
                        line2[0] = -100;
                    }
                }
            }
        }
    }

    public static NeuralNet loadNNFromFile(String fileName){
                NeuralNet nn = null;
        try
        {
            FileInputStream fileIn = new FileInputStream(fileName);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            nn = (NeuralNet) in.readObject();
            in.close();
            fileIn.close();
        }catch(IOException i)
        {
            i.printStackTrace();
            return nn;
        }catch(ClassNotFoundException c)
        {
            System.out.println("Employee class not found");
            c.printStackTrace();
        }
        System.out.println("nn created");
        return nn;
    }
}
