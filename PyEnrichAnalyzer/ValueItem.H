//
// Created by Halberg, Spencer on 3/18/25.
//

#ifndef VALUEITEM_H
#define VALUEITEM_H
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;


using namespace std;

class ValueItem {
	public:

	ValueItem(string, string, double , double , int ,
	          int , int , int , double , string);

	ValueItem(string, string, double , double , int ,
		  int , int , int , double);

	ValueItem();

	int setSubGraphName(string);
	int setTermName(string);
	int setUncorrPval(double);
	int setCorrPval(double);
	int setTotal(int);
	int setN1(int);
	int setT(int);
	int setHitCnt(int);
	int setFoldenr(double);
	int setHitString(string);

	string getSubGraphName();
	string getTermName();
	double getUnCorrPval();
	double getCorrPval();
	int getTotal();
	int getN1();
	int getT();
	int getHitCnt();
	double getFoldenr();
	string getHitString();

	py::dict valueItemToPythonDict();


	private:
	string SGName;
	string termName;
	double uncorrPval;
	double corrPval;
	int total;
	int n1;
	int t;
	int hitCnt;
	double foldenr;
	string hitString;
};



#endif //VALUEITEM_H
