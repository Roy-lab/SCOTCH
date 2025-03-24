#ifndef _GOTERM
#define _GOTERM

#include <fstream>
#include <string>
#include <map>

using namespace std;

class GOTerm
{
	public:
		GOTerm();
		~GOTerm();
		//Although we are saying tissue, really we mean tissue and stage development
		int setGOName(const char*);
		const char* getGOName();
		
		int addMemberGene(const char*);
		map<string,int>& getMemberGenes();
		int getMemberCnt();
		bool isMember(const string&);
		bool isMember(const char*);

	private:
		string goTerm;
		map<string,int> memberGenes;
};
#endif
