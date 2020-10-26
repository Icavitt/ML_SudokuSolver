package MyProjectGroup;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.InputStream;

/**
 * Created by Ian on 4/14/2016.
 */
class TextFieldStreamer extends InputStream implements KeyListener {

    private JTextField tf;
    private String str = null;
    private int pos = 0;

    public TextFieldStreamer(JTextField jtf) {
        tf = jtf;
    }

    public void actionPerformed(ActionEvent e) {
        str = tf.getText() + "\n";
        pos = 0;
        tf.setText("");
        synchronized (this) {
            //maybe this should only notify() as multiple threads may
            //be waiting for input and they would now race for input
            this.notifyAll();
        }
    }

    @Override
    public int read() {
        //test if the available input has reached its end
        //and the EOS should be returned
        if(str != null && pos == str.length()){
            str =null;
            //this is supposed to return -1 on "end of stream"
            //but I'm having a hard time locating the constant
            return java.io.StreamTokenizer.TT_EOF;
        }
        //no input available, block until more is available because that's
        //the behavior specified in the Javadocs
        while (str == null || pos >= str.length()) {
            try {
                //according to the docs read() should block until new input is available
                synchronized (this) {
                    this.wait();
                }
            } catch (InterruptedException ex) {
                ex.printStackTrace();
            }
        }
        //read an additional character, return it and increment the index
        return str.charAt(pos++);
    }

    public void keyTyped(KeyEvent e) {

    }

    public void keyPressed(KeyEvent e) {
        if (e.getKeyCode() == e.VK_ENTER){
//            int endpos = tf.getCaret().getMark();
//            int startpos = tf.getText().substring(0, endpos-1).lastIndexOf('\n')+1;
            str = tf.getText() + "\n";
            System.out.print(str);
            pos = 0;
            tf.setText("");
        }
    }

    public void keyReleased(KeyEvent e) {

    }
}