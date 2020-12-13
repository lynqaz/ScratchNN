#include "tictactoe.h"
#include "AI.h"


std::vector<int> extent = { 10,42,8,1 };
double learningRate = 0.1;
std::vector<std::vector<double>> boardPositions, bestMoves;
int move;
AI NN(extent);
tictactoe ttt;
void selfplay(int);
int TestNN(int depth, std::vector<int> board, int currentplayer, double& scoreReturn);
int findBestMove(std::vector<int> board, bool training, double currentplayer, double& scoreReturn, std::vector<int> lastmoves1 = {}, std::vector<int> lastmoves2 = {});
int occurencesOfNumberInVector(int, std::vector<int>);
double sigmoid(double x) {
	return 1 / (1 + exp(-x * .2));
}

int main() {
	int neuralNetwork = 0, randomPlayer = 0, ties = 0;
	selfplay(10000);
	ttt.reset();
	ttt.printboard();
	double score;
	for (int s = 0; s < 10; s++) {
		for (int i = 0; i < 9; ++i) {
			if (i % 2 == 1) {
				move = ttt.randommove();
				//std::cin >> move;
				i -= (ttt.move(move)) ? 0 : 1;
			}
			else {
				i -= (ttt.move(findBestMove(ttt.getoriginalboard(), true, (double)ttt.getcurrentplayer() - 1, score))) ? 0 : 1;
			}
			int winner = ttt.getwinner();
			if (winner != 0) {
				ttt.printboard();
				if (winner == 1) {
					neuralNetwork += 1;
				}
				else {
					randomPlayer += 1;
				}
				std::cout << "\nPLAYER " << winner << " WINS!!!\n\n";
				ttt.reset();
				i = -1;
			}
			ttt.printboard();
		}
		if (ttt.getwinner() == 0) {
			ties += 1;
		}
		ttt.reset();
	}
	std::cout << "Neural Network wins :  " << neuralNetwork << std::endl;
	std::cout << "Random Player wins :   " << randomPlayer << std::endl;
	std::cout << "ties :                 " << ties << std::endl;
	std::cin >> move;
	return 0;
}

void selfplay(int matches) {
	//counts the number of occurences a particular move in the most recent 100 moves;
	int in = 0;
	int x = 0, o = 0;
	double match = 1;
	do {
		std::vector<int> lastmoves1, lastmoves2;
		for (int i = 0; i < matches; ++i) {
			std::vector<std::vector<double>> inputs = {};
			ttt.reset();
			for (int move = 0; move < 9; ++move) {
				//check whether last move was a winning one
				if (ttt.getwinner() != 0) {
					if (ttt.getwinner() == 1) {
						x += 1;
					}
					else{
						o += 1;
					}
					break;
				}
				std::vector<int> board = ttt.getoriginalboard();
				double score;
				//let the NN find the best move
				int m = findBestMove(board, true, (double)ttt.getcurrentplayer() - 1, score, lastmoves1, lastmoves2);
				if (move % 2 == 0) {
					lastmoves1.push_back(m);
					if (lastmoves1.size() > 100) { lastmoves1.erase(lastmoves1.begin()); }
				}
				else {
					lastmoves2.push_back(m);
					if (lastmoves2.size() > 100) { lastmoves2.erase(lastmoves2.begin()); }
				}
				ttt.move(m);
				std::vector<double> splitboard = ttt.getboard();
				inputs.push_back(splitboard);
			}
			ttt.printboard();
			double score = ttt.getscore();
			
			for (int index = 0; index < (signed)inputs.size(); ++index) {
				if (index % 2 == 0 && score == .5) {
					score = 1;
				}
				else if (index % 2 == 1 && score == .5) {
					score = 0;
				}
				NN.selfplay({ inputs[index] }, { { score } }, learningRate);
			}
			std::cout << "                                                   X : " << x << "     O : " << o << std::endl;
		}
		NN.getloss({ { 0,0,-10,0,-10,10,10,10,0,-10 } });
		NN.getloss({ { 0,0,10,0,10,-10,0,-10,0,10 } });
		std::cin >> in;
	} while (in == 1);
}
int findBestMove(std::vector<int> board, bool training, double currentplayer, double& scoreReturn, std::vector<int> lastmoves1, std::vector<int> lastmoves2) {
	std::vector<std::vector<double>> outputs, inputs;
	//each possible move gets rated by the NN, depending on the Player will the highest or the lowest be the next move
	int nextmove;
	double bestscore = currentplayer;
	for (int move = 0; move < 9; ++move) {
		if (board[move] == 0) {
			board[move] = ttt.getcurrentplayer();
			//create splitboard for NN
			std::vector<double> splitboard = {};
			for (int i = 0; i < 9; ++i) {
				if (board[i] == 1) {
					splitboard.push_back(10);
				}
				else if (board[i] == 2) {
					splitboard.push_back(-10);
				}
				else {
					splitboard.push_back(0);
				}
			}
			if (ttt.getcurrentplayer() == 1) {
				splitboard.push_back(10);
			} else{
				splitboard.push_back(-10);
			}
			//run splitboard as input through NN and store output in outputs
			outputs.push_back(NN.run(splitboard));
			//occurencesOfNumberInVector makes sure every so often a new move will be explored, even though the NN predicts another move to be better
			if (training) {
				if (outputs[move][0] * (-sigmoid((double)occurencesOfNumberInVector(move, lastmoves1)) + 1.5) > bestscore && ttt.getcurrentplayer() == 1) {
					bestscore = outputs[move][0] * (-sigmoid((double)occurencesOfNumberInVector(move, lastmoves1)) + 1.5);
					nextmove = move;
				}
				else if (outputs[move][0] * (sigmoid((double)occurencesOfNumberInVector(move, lastmoves2))) < bestscore && ttt.getcurrentplayer() == 2) {
					bestscore = outputs[move][0] * sigmoid((double)occurencesOfNumberInVector(move, lastmoves2));
					nextmove = move;
				}
			}
			else {
				if (outputs[move][0] > bestscore && ttt.getcurrentplayer() == 1) {
					bestscore = outputs[move][0];
					nextmove = move;
				}
				else if (outputs[move][0] < bestscore && ttt.getcurrentplayer() == 2) {
					bestscore = outputs[move][0];
					nextmove = move;
				}
			}
			board[move] = 0;
		}else{ 
			//this move is not possible and therefore should not be picked
			outputs.push_back({ {0.5} });
		}
	}
	scoreReturn = bestscore;
	return nextmove;
}
int occurencesOfNumberInVector(int number, std::vector<int> vector) {
	int occurences = 0;
	for (int i = 0; i < (signed)vector.size(); ++i) {
		if (vector[i] == number) {
			occurences += 1;
		}
	}
	return occurences;
}

int TestNN(int depth, std::vector<int> board, int currentplayer, double& scoreReturn) {
	//function that helps the NN see future boardstates
	int bestmove = 0;
	double bestscore = (double)currentplayer - 1, score = (double)currentplayer - 1;
	if (depth == 0) {
		return bestmove = findBestMove(board, false, bestscore, scoreReturn);
	}
	for (int i = 0; i < 9; i++) {
		if (board[i] == 0) {
			board[i] = currentplayer;
			TestNN(depth - 1, board, -currentplayer + 3, score);
			if (currentplayer == 1) {
				if (score > bestscore) {
					bestscore = score;
					bestmove = i;
				}
			}
			else {
				if (score < bestscore) {
					bestscore = score;
					bestmove = i;
				}
			}
			board[i] = 0;
		}
	}
	scoreReturn = bestscore;
	return bestmove;
}