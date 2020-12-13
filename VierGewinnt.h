#pragma once
#include <vector>
#include <iostream>
#include <algorithm>

class VierGewinnt
{
public:

	std::vector<double> getboard();
	std::vector<int> getoriginalboard();
	std::vector<int> getpossiblemoves();
	int getcurrentplayer();
	int getwinner();
	int getmoves();
	double getscore();
	int randommove();
	bool move(int);
	void printboard();
	void reset();

private:

	std::vector<int> m_board = { 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0 };
	int m_currentPlayer = 1;
	int m_moves = 0;

	void changeplayer();
	int checkboard(std::vector<int>);
};

