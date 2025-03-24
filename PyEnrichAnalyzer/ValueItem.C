//
// Created by Halberg, Spencer on 3/18/25.
//

#include "ValueItem.H"

ValueItem::ValueItem(string sg_name,string term_name, double uncorr_pval, double corr_pval, int total,
		int n1, int T, int hit_cnt, double foldenr,string hit_string)
		: SGName(sg_name),
		  termName(term_name),
		  uncorrPval(uncorr_pval),
		  corrPval(corr_pval),
		  total(total),
		  n1(n1),
		  t(T),
		  hitCnt(hit_cnt),
		  foldenr(foldenr),
		  hitString(hit_string)
{
}

ValueItem::ValueItem(string sg_name,string term_name, double uncorr_pval, double corr_pval, int total,
		int n1, int T, int hit_cnt, double foldenr)
		: SGName(sg_name),
		  termName(term_name),
		  uncorrPval(uncorr_pval),
		  corrPval(corr_pval),
		  total(total),
		  n1(n1),
		  t(T),
		  hitCnt(hit_cnt),
		  foldenr(foldenr),
		  hitString("")
{
}


ValueItem::ValueItem()
	: SGName(""),
	  termName(""),
	  uncorrPval(0.0),
	  corrPval(0.0),
	  total(0),
	  n1(0),
	  t(0),
	  hitCnt(0),
	  foldenr(0.0),
	  hitString("")
{
}


int
ValueItem::setSubGraphName(string name)
{
	SGName=name;
	return 0;
}

int
ValueItem::setTermName(string term)
{
	termName=term;
	return 0;
}

int
ValueItem::setUncorrPval(double pval)
{
	uncorrPval = pval;
	return 0;
}

int
ValueItem::setCorrPval(double pval)
{
	corrPval = pval;
	return 0;
}


int
ValueItem::setTotal(int tot)
{
	total = tot;
	return 0;
}


int
ValueItem::setN1(int n)
{
	n1 = n;
	return 0;
}


int
ValueItem::setT(int n)
{
	t = n;
	return 0;
}


int
ValueItem::setHitCnt(int hit)
{
	hitCnt = hit;
	return 0;
}

int
ValueItem::setFoldenr(double fe)
{
	foldenr =  fe;
	return 0;
}

int
ValueItem::setHitString(string hits)
{
	hitString = hits;
	return 0;
}

string
ValueItem::getSubGraphName()
{
	return SGName;
}

string
ValueItem::getTermName()
{
	return termName;
}

double
ValueItem::getUnCorrPval()
{
	return uncorrPval;
}

double
ValueItem::getCorrPval()
{
	return corrPval;
}

int
ValueItem::getTotal()
{
	return total;
}

int
ValueItem::getN1()
{
	return n1;
}

int
ValueItem::getT()
{
	return t;
}


int
ValueItem::getHitCnt()
{
	return hitCnt;
}


double
ValueItem::getFoldenr()
{
	return foldenr;
}

string
ValueItem::getHitString()
{
	return hitString;	
}

py::dict ValueItem::valueItemToPythonDict()
{
	py::dict dict;

	dict["SubGraphName"] = py::str(SGName);
	dict["TermName"] = py::str(termName);
	dict["UncorrPval"] = uncorrPval;
	dict["CorrPval"] = corrPval;
	dict["Total"] = total;
	dict["N1"] = n1;
	dict["t"] = t;
	dict["HitCnt"] = hitCnt;
	dict["Foldenr"] = foldenr;
	dict["HitString"] = py::str(hitString);

	return dict;
}











