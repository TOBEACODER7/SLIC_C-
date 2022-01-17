#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <string.h>
#include <vector>

using namespace std;
using namespace cv;
int getlen(Point p1, Point p2, int s, Mat img);
struct cenpoint
{
	Point p;
	int l;
	int a;
	int b;
};
int main()
{
	Mat img = imread("test3.jpg");
	Mat img_lab;
	resize(img, img, Size(540, 540));

	//将图像由bgr转为lab
	cvtColor(img, img_lab, COLOR_BGR2Lab);

	//确定簇数、步长
	int y = img_lab.rows;
	int x = img_lab.cols;
	int k = 100;
	int s = sqrt(x*y/k);
	vector<cenpoint> cenpot;

	//初始化聚类中心
	for (int j = 0; j < 10; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			cenpoint pot;
			pot.p = Point2i(i * s + s / 2-1, j * s + s / 2-1);
			Vec3b p1 = img_lab.at<Vec3b>(pot.p);
			pot.l = p1[0];
			pot.a = p1[1];
			pot.b = p1[2];
			cenpot.push_back(pot);
		}
	}

	
	//优化聚类中心
	for (int i = 0; i < cenpot.size(); i++)
	{
		int g[3][3];
		memset(g, 1000, sizeof(g));
		int min = 400;
		int min_x = cenpot[i].p.x;
		int min_y = cenpot[i].p.y;
		for (int x = cenpot[i].p.x - 1; x < cenpot[i].p.x + 1; x++)
		{
			for (int y = cenpot[i].p.y - 1; y < cenpot[i].p.y + 1; y++)
			{
				Point p1 = Point2i(x, y);
				Point p2 = Point2i(x+1,y);
				Point p3 = Point2i(x, y+1);
				Vec3i p1_lab = img_lab.at<Vec3i>(p1);
				Vec3i p2_lab = img_lab.at<Vec3i>(p2);
				Vec3i p3_lab = img_lab.at<Vec3i>(p3);
				g[x - cenpot[i].p.x + 2][y - cenpot[i].p.y + 2] = abs(p1_lab[0] - p2_lab[0]) + abs(p1_lab[0] - p3_lab[0]);
				if (g[x - cenpot[i].p.x + 2][y - cenpot[i].p.y + 2] < min)
				{
					min = g[x - cenpot[i].p.x + 2][y - cenpot[i].p.y + 2];
					min_x = x;
					min_y = y;
				}
			}
		}
		cenpot[i].p.x = min_x;
		cenpot[i].p.y = min_y;
		Vec3i p1 = img_lab.at<Vec3i>(cenpot[i].p);
		cenpot[i].l = p1[0];
		cenpot[i].a = p1[1];
		cenpot[i].b = p1[2];
	}

	Mat label = -1 * Mat::ones(img_lab.size(), CV_32S);
	Mat len = -1 * Mat::ones(img_lab.size(), CV_32F);
	//Mat SWin;/*2S x 2S的滑动窗口*/
	Point p1, p2;
	Vec3d p1_lab, p2_lab;
	//聚类过程
	for (int idx = 0; idx < 10; idx++)
	{

		for (int c = 0; c < cenpot.size(); c++)
		{
			p1 = cenpot[c].p;

			for (int y = p1.y - s; y <= p1.y + s; y++)
			{

				if (y < 0)
					y = 0;
				if (y >= img_lab.rows)
					break;
				for (int x = p1.x - s; x <= p1.x + s; x++)
				{

					if (x < 0)
						x = 0;
					if(x>=img_lab.cols)
						break;
					p2 = Point2i(x, y);
					float d = getlen(p1, p2,s,img_lab);
					if (len.at<float>(p2) > d || len.at<float>(p2) == -1 || label.at<int>(p2) == -1)
					{
						len.at<float>(p2) = d;
						label.at<int>(p2) = c;
					}
				}
			}
		}

		//更新聚类中心
		for (int c = 0; c < cenpot.size(); c++)
		{
			cenpoint ans;
			ans.p.x = 0;
			ans.p.y = 0;
			ans.l = 0;
			ans.a = 0;
			ans.b = 0;
			int num=0;
			for (int j = 0; j < label.rows; j++)
			{
				for (int i = 0; i < label.cols; i++)
				{
					if (label.at<int>(Point2i(i, j)) == c)
					{
						num++;
						ans.p.x += i;
						ans.p.y += j;
						Vec3b px = img_lab.at<Vec3b>(Point2i(i, j));
						ans.l += px[0];
						ans.a += px[1];
						ans.b += px[2];
					}
				}
			}

			if (num == 0)
				continue;
			cenpot[c].p.x = round(ans.p.x / num);
			cenpot[c].p.y = round(ans.p.y / num);
			cenpot[c].l = round(ans.l / num);
			cenpot[c].a = round(ans.a / num);
			cenpot[c].b = round(ans.b / num);
			
			
		}
		
	}
	//结果展示的一些函数
	Mat copy = img_lab.clone();
	for (int c = 0; c < 100; c++)
	{
		circle(copy, cenpot[c].p, 1, Scalar(0, 0, 255), 1);
	}
	imshow("超像素示意图", copy);
	Mat img_ans = img_lab.clone();
	for (int i = 0; i < img_lab.cols; i++)
	{
		for (int j = 0; j < img_lab.rows; j++)
		{
			int idx = label.at<int>(Point2i(i,j));

			img_ans.at<Vec3b>(j, i)[0] = cenpot[idx].l;
			img_ans.at<Vec3b>(j, i)[1] = cenpot[idx].a;
			img_ans.at<Vec3b>(j, i)[2] = cenpot[idx].b;
		}
	}
	cvtColor(img_ans, img_ans, COLOR_Lab2BGR);
	imshow("分割图", img_ans);

	Mat img_edge = img_lab.clone();
	for (int i = 0; i < img_edge.cols; i++)
	{
		for (int j = 0; j < img_edge.rows; j++)
		{
			int idx = label.at<int>(Point2i(i, j));
			img_edge.at<Vec3b>(j, i)[0] = cenpot[idx].l;
			img_edge.at<Vec3b>(j, i)[1] = cenpot[idx].a;
			img_edge.at<Vec3b>(j, i)[2] = cenpot[idx].b;
		}
	}
	cvtColor(img_edge, img_edge, COLOR_Lab2BGR);
	for (int i = 0; i < img_edge.cols; i++)
	{
		for (int j = 0; j < img_edge.rows; j++)
		{
			int idx = label.at<int>(Point2i(i, j));
			if (i - 1 < 0 || j - 1 < 0 || i + 1 >= img_edge.cols || j + 1 >= img_edge.rows)
				continue;
			if (idx != label.at<int>(Point2i(i - 1, j)) || idx != label.at<int>(Point2i(i + 1, j)) || idx != label.at<int>(Point2i(i, j - 1)) || idx != label.at<int>(Point2i(i, j + 1)))
				img_edge.at<Vec3b>(j,i)[0] = img_edge.at<Vec3b>(j,i)[1] = img_edge.at<Vec3b>(j,i)[2] = 0;
		}
	}
	imshow("边界", img_edge);


	waitKey(0);
	return 0;

}
//计算五维向量距离的函数
int getlen(Point p1, Point p2,int s,Mat img)
{
	Vec3b p1_lab = img.at<Vec3b>(p1);
	Vec3b p2_lab = img.at<Vec3b>(p2);
	int l = p1_lab[0] - p2_lab[0];
	int a = p1_lab[1] - p2_lab[1];
	int b = p1_lab[2] - p2_lab[2];
	int x = p1.x - p2.x;
	int y = p2.y - p2.y;
	int dc = l * l + a * a + b * b;
	int ds = x * x + y * y;
	int m = 10;
	return sqrt(dc / m / m + ds / s / s);
}
