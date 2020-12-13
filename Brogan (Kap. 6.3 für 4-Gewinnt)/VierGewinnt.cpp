#include "VierGewinnt.h"


std::vector<double> VierGewinnt::getboard() {
	std::vector<double> board;
	for (int i = 0; i < 42; ++i) {
		if (m_board[i] == 1) {
			board.push_back(10);
		}
		else if (m_board[i] == 2) {
			board.push_back(-10);
		}
		else {
			board.push_back(0);
		}
	}
	int cp = getcurrentplayer();
	board.push_back(((signed)cp * 2 - 3) * 10);
	return board;
};
std::vector<int> VierGewinnt::getoriginalboard() {
	return m_board;
}
std::vector<int> VierGewinnt::getpossiblemoves() {
	//empty square = 1, occupied = 0
	std::vector<int> possibleMoves(7, 0);
	for (int i = 0; i < 7; ++i) {
		for (int col = 5; col >= 0; --col) {
			if (m_board[7 * col + i] == 0) {
				possibleMoves[i] = 1;
			}
		}
	}
	return possibleMoves;
};
bool VierGewinnt::move(int row) {
	//set square to currentplayer
	for (int col = 5; col >= 0; --col) {
		if (m_board[7 * col + row] == 0) {
			m_board[7 * col + row] = m_currentPlayer;
			m_currentPlayer = -m_currentPlayer + 3;
			return true;
		}
	}
	std::cout << "this move is not allowed!\n\n";
	return false;
};
int VierGewinnt::getwinner() {
	if (checkboard(m_board) == -m_currentPlayer + 3) {
		return -m_currentPlayer + 3;
	}
	return 0;
}
double VierGewinnt::getscore() {
	if (checkboard(m_board) == 1) {
		return 1;
	}
	else if (checkboard(m_board) == 2) {
		return 0;
	}
	else {
		return 0.5;
	}
}
void VierGewinnt::reset() {
	m_board = { 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0 };
	m_currentPlayer = 1;
	m_moves = 0;
};

void VierGewinnt::changeplayer() {
	m_currentPlayer = -m_currentPlayer + 3;
};

int VierGewinnt::checkboard(std::vector<int> board) {
	//separating the board into two boards, one for player one and one for player two
	std::vector<std::vector<std::vector<int>>> separatedBoard;
	for (int i = 1; i < 3; ++i) {
		std::vector<std::vector<int>> sepBoardTemp;
		for (int col = 0; col < 6; ++col) {
			std::vector<int> colVecTemp;
			for (int row = 0; row < 7; ++row) {
				colVecTemp.push_back((board[col * 7 + row] == i) ? 1 : 0);
			}
			sepBoardTemp.push_back(colVecTemp);
		}
		separatedBoard.push_back(sepBoardTemp);
	}
	
	//columns and rows

	// counts the fields, that are occupied by a player, adjacent to one another and in the same row or column
	int streak = 0;
	for (int i = 0; i < 2; ++i) {
		for (int col = 0; col < 6; ++col) {
			streak = 0;
			for (int row = 0; row < 7; ++row) {
				if (separatedBoard[i][col][row] == 1) {
					streak += 1;
					if (streak == 4) {
						return i + 1;
					}
				}
				else {
					streak = 0;
				}
			}
		}
	}
	for (int i = 0; i < 2; ++i) {
		for (int row = 0; row < 7; ++row) {
			streak = 0;
			for (int col = 0; col < 6; ++col) {
				if (separatedBoard[i][col][row] == 1) {
					streak += 1;
					if (streak == 4) {
						return i + 1;
					}
				}
				else {
					streak = 0;
				}
			}
		}
	}

	//diagonals

	for (int i = 0; i < 2; ++i) {
		int row = 3;
		int col = 0;
		for (int a = 0; a < 4; ++a) {
			streak = 0;
			int c = col, r = row;
			for (int b = 0; b <= row; ++b) {
				if (separatedBoard[i][c][r] == 1) {
					streak += 1;
					if (streak == 4) {
						return i + 1;
					}
				}
				else {
					streak = 0;
				}
				r -= 1;
				if (r > 0) {
					c += 1;
				}
			}
			row += 1;
		}

		row = 3;
		col = 5;
		for (int a = 0; a < 3; ++a) {
			streak = 0;
			int c = col, r = row;
			for (int b = 0; b < 7 - row; ++b) {
				if (separatedBoard[i][c][r] == 1) {
					streak += 1;
					if (streak == 4) {
						return i + 1;
					}
				}
				else {
					streak = 0;
				}
				r += 1;
				c -= 1;
			}
			row -= 1;
		}

		row = 3;
		col = 0;
		for (int a = 0; a < 4; ++a) {
			streak = 0;
			int c = col, r = row;
			for (int b = 0; b < 7 - row; ++b) {
				if (separatedBoard[i][c][r] == 1) {
					streak += 1;
					if (streak == 4) {
						return i + 1;
					}
				}
				else {
					streak = 0;
				}
				r += 1;
				if (r < 6) {
					c += 1;
				}
			}
			row -= 1;
		}

		row = 3;
		col = 5;
		for (int a = 0; a < 3; ++a) {
			streak = 0;
			int c = col, r = row;
			for (int b = 0; b <= row; ++b) {
				if (separatedBoard[i][c][r] == 1) {
					streak += 1;
					if (streak == 4) {
						return i + 1;
					}
				}
				else {
					streak = 0;
				}
				r -= 1;
				c -= 1;
			}
			row += 1;
		}
	}
	//neither player has 3 in a row
	return false;
};
void VierGewinnt::printboard() {

	std::cout << "-----------------------------------\nTIC TAC TOE :\nmove number: " << m_moves << "\ncurrent player : " << m_currentPlayer << std::endl << std::endl;
	for (int i = 0; i < 42; ++i) {
		std::string out;
		if (m_board[i] == 1) {
			out = "X";
		}
		else if (m_board[i] == 2) {
			out = "O";
		}
		else {
			out = "_";
		}
		std::cout << out << " ";
		if ((i + 1) % 7 == 0) {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl << std::endl;
};
int VierGewinnt::getcurrentplayer() {
	return m_currentPlayer;
};
int VierGewinnt::getmoves() {
	return m_moves;
}
int VierGewinnt::randommove() {
	std::vector<int> posMoves, v;
	v = getpossiblemoves();
	for (int i = 0; i < 7; ++i) {
		if (v[i] == 1) { posMoves.push_back(i); }
	}
	return posMoves[rand() % (signed)posMoves.size()];
};
