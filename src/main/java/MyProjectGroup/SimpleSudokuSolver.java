package MyProjectGroup;

/**
 * Created by Ian on 4/8/2016.
 */
public class SimpleSudokuSolver {
            private static void solve(int[][] board, int ind){
            int size = board.length;
            if(ind == size*size){
                printSolution(board);
                System.out.println();
            }
            else{
                int row = ind / size;
                int col = ind % size;
                if(board[row][col] != 0){
                    solve(board, ind+1);
                }
                else{
                    for(int i = 1; i <= 9; i++){
                        if(isConsistent(board, row, col, i)){
                            board[row][col] = i;
                            solve(board,ind+1);
                            board[row][col] = 0;
                        }
                    }
                }
//                System.out.println("no solution...?");
            }

        }
        private static void printSolution(int[][] board){
            System.out.println(" _ _ _ _ _ _ _ _ _ ");
            for(int[] row : board){
                for(int entry : row){
                    System.out.print("|" + entry);
                }
                System.out.print("|\n");
            }
            System.out.print(" - - - - - - - - - \n");
        }

        private static boolean isConsistent(int[][] board, int row, int col, int c) {
            int size = board.length;
            int subesize = 3;
            for(int i = 0; i < size; i++){
                if(board[row][i] == c){
                    return false;
                }
                if(board[i][col] == c){
                    return false;
                }
            }
            int rowStart = row - row % subesize;
            int colStart = col - col % subesize;

            for(int m = 0; m < subesize; m++){
                for(int k = 0; k < subesize; k++){
                    if(board[rowStart + k][colStart + m] == c){
                        return false;
                    }
                }
            }
            return true;
        }

        public static void solve(int[][] sudoku){
            solve(sudoku,0);
        }
}
