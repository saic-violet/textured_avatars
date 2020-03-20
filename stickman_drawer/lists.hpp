#pragma once
#include <vector>

#define pose_edges_num_07 18
#define pose_edges_num_07_earneck 20
#define pose_edges_num_12 24
#define pose_edges_num_12_earneck 26
#define hand_edges_num 20
#define face_inner_edges_num 39
#define face_pupils_edges_num 2
#define face_contour_edges_num 16
#define face_inner_mouth_edges_num 8

#define pose_seq_num 18
#define hand_seq_num 5
#define face_seq_num 6

using index_pair = int[2];

struct ColorT
{
	uint8_t r,g,b;
};

static const index_pair pose_edge_list_07[pose_edges_num_07] = {
	{ 0,  1}, { 1, 15}, {15, 16}, { 1, 17}, {17, 18},			//head
	{ 0,  2}, 													//body
	{ 0,  9}, { 9, 10}, {10, 11},								//right arm
	{ 0,  3}, { 3,  4}, { 4,  5},								//left arm
	{ 2,  6}, { 6,  7}, { 7,  8},								//right leg
	{ 2, 12}, {12, 13}, {13, 14}								//left leg
};

static const index_pair pose_edge_list_07_earneck[pose_edges_num_07_earneck] = {
		{ 0,  1}, { 1, 15}, {15, 16}, { 1, 17}, {17, 18}, {0, 18},	{0,17},		//head
		{ 0,  2}, 													//body
		{ 0,  9}, { 9, 10}, {10, 11},								//right arm
		{ 0,  3}, { 3,  4}, { 4,  5},								//left arm
		{ 2,  6}, { 6,  7}, { 7,  8},								//right leg
		{ 2, 12}, {12, 13}, {13, 14}								//left leg
};

static const index_pair pose_edge_list_12[pose_edges_num_12] = {
{17, 15}, {15,  0}, { 0, 16}, {16, 18}, { 0,  1},           // head
{ 1,  8},                                                   // body
{ 1,  2}, { 2,  3}, { 3,  4},                               // right arm
{ 1,  5}, { 5,  6}, { 6,  7},                               // left arm
{ 8,  9}, { 9, 10}, {10, 11}, {11, 24}, {11, 22}, {22, 23}, // right leg
{ 8, 12}, {12, 13}, {13, 14}, {14, 21}, {14, 19}, {19, 20}  // left leg
};

static const index_pair pose_edge_list_12_earneck[pose_edges_num_12_earneck] = {
        {17, 15}, {15,  0}, { 0, 16}, {16, 18}, { 0,  1}, {1,17}, {1,18},          // head
        { 1,  8},                                                   // body
        { 1,  2}, { 2,  3}, { 3,  4},                               // right arm
        { 1,  5}, { 5,  6}, { 6,  7},                               // left arm
        { 8,  9}, { 9, 10}, {10, 11}, {11, 24}, {11, 22}, {22, 23}, // right leg
        { 8, 12}, {12, 13}, {13, 14}, {14, 21}, {14, 19}, {19, 20}  // left leg
};

//static const ColorT pose_color_list [pose_edges_num] = {
//	{153,  0,153}, {153,  0,102}, {102,  0,153}, { 51,  0,153}, {153,  0, 51},
//	{153,  0,  0},
//	{153, 51,  0}, {153,102,  0}, {153,153,  0},
//	{102,153,  0}, { 51,153,  0}, {  0,153,  0},
//	{  0,153, 51}, {  0,153,102}, {  0,153,153},
//	{  0,102,153}, {  0, 51,153}, {  0,  0,153}
//};

//if rand_color:
//    pose_color_list = {np.random.randint(0, 256 ,size=3) for i in range(len(pose_color_list))}

//### hand
static const index_pair hand_edge_list[hand_edges_num] = {
	{0,1},{1,2},{2,3},{3,4},
	{0,5},{5,6},{6,7},{7,8},
	{0,9},{9,10},{10,11},{11,12},
	{0,13},{13,14},{14,15},{15,16},
	{0,17},{17,18},{18,19},{19,20}
};




static const ColorT hand_color_list [hand_edges_num] = {
	{204,0,0}, {204,0,0}, {204,0,0}, {204,0,0},
	{163,204,0}, {163,204,0}, {163,204,0}, {163,204,0},
	{0,204,82}, {0,204,82}, {0,204,82}, {0,204,82},
	{0,82,204}, {0,82,204}, {0,82,204}, {0,82,204},
	{163,0,204}, {163,0,204}, {163,0,204}, {163,0,204}
};


//if rand_color:
//   hand_color_list = {np.random.randint(0, 256 ,size=3) for i in range(len(hand_color_list))}

//### face        
static const index_pair face_inner_edge_list[face_inner_edges_num] = {
{17,18},{18,19},{19,20},{20,21}, // left eyebrow   4
{22,23},{23,24},{24,25},{25,26}, // right eyebrow  4
{27,28},{28,29},{29,30},		 // nose           7
{31,32},{32,33},{33,34},{34,35},
{36,37},{37,38},{38,39},		 // left eye       6
{39,40},{40,41},{41,36},
{42,43},{43,44},{44,45},		 // right eye      6
{45,46},{46,47},{47,42},
{48,49},{49,50},{50,51},{51,52}, // outer mouth   12
{52,53},{53,54},{54,55},{55,56},
{56,57},{57,58},{58,59},{59,48}, 
};

static const index_pair face_pupils_edge_list[face_pupils_edges_num] = {
{68,68},{69,69}					 // face pupils    2
};

static const index_pair face_contour_edge_list[face_contour_edges_num] = {
{ 0, 1},{ 1, 2},{ 2, 3},{ 3, 4}, // face contour  16
{ 4, 5},{ 5, 6},{ 6, 7},{ 7, 8},
{ 8, 9},{ 9,10},{10,11},{11,12},
{12,13},{13,14},{14,15},{15,16}
};

static const index_pair face_inner_mouth_edge_list[face_inner_mouth_edges_num] = {
{60,61},{61,62},{62,63},{63,64}, // inner mouth    8
{64,65},{65,66},{66,67},{67,60}	 
};

static const uint8_t face_inner_color_list[face_inner_edges_num][1] = {
{255},{255},{255},{255},{255},{255},{255},{255},{255},{255}, // 10
{255},{255},{255},{255},{255},{255},{255},{255},{255},{255},
{255},{255},{255},{255},{255},{255},{255},{255},{255},{255},
{255},{255},{255},{255},{255},{255},{255},{255},{255}
};

static const uint8_t face_pupils_color_list[face_pupils_edges_num][1] = {
{255},{255}
};

static const uint8_t face_contour_color_list[face_contour_edges_num][1] = {
{255},{255},{255},{255},{255},{255},{255},{255},{255},{255}, // 10
{255},{255},{255},{255},{255},{255}
};

static const uint8_t face_inner_mouth_color_list[face_inner_mouth_edges_num][1] = {
{255},{255},{255},{255},{255},{255},{255},{255}
};

static const int face_inner_list_lengths[6]={4,4,7,6,6,12};
static const int face_pupils_list_lengths[6]={1,1};
static const int face_contour_list_lengths[1]={16};
static const int face_inner_mouth_list_lengths[1]={8};

static const int hand_list_lengths[5]={4,4,4,4,4};
static const int pose_list_lengths[18]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

static int hand_channels[hand_edges_num]={19,19,19,19,
										  20,20,20,20,
										  21,21,21,21,
										  22,22,22,22,
										  23,23,23,23};


