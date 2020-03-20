#include <iostream>
#include "lists.hpp"
#include <math.h>
#include <algorithm>
#include <array>

//#define GLOBAL_Z_MIN 100
//#define GLOBAL_Z_MAX 500


using namespace std;

struct PointT {
    float x, y;
};
struct LineT {
    float x1, x2, y1, y2, z1, z2;
};

class StickmanData_C {
    int _n = 4;
//	int _len[4]={19,70,21,21};
    int _shift3[4] = {0, 19 * 3, 19 * 3 + 70 * 3, 19 * 3 + 70 * 3 + 21 * 3};
    int _shift4[4] = {0, 19 * 4, 19 * 4 + 70 * 4, 19 * 4 + 70 * 4 + 21 * 4};

    std::string _fields[4] = {"pose_keypoints_3d", "face_keypoints_3d", "hand_left_keypoints_3d",
                              "hand_right_keypoints_3d"};
    std::vector<bool> _valid;
    std::vector<float> _extracted_data;
    float _faceMean = 0.0f;
    std::vector<std::vector<int>> pose_edge_list;
    int pose_edges_num;
    int face_edges_num;
    std::vector<std::vector<int>> face_edge_list;
    int hand_map_num, face_map_num;
public:
	std::vector<float> _data;
    bool is_07, is_separate_hands, is_separate_face, is_earneck, draw_face_pupils, draw_face_contour, draw_face_inner_mouth;
    int global_z_min, global_z_max;
	StickmanData_C(
		bool is_07=true, bool is_separate_hands=false, bool is_separate_face=false, bool is_earneck=false,
		bool draw_face_pupils=false, bool draw_face_contour=false, bool draw_face_inner_mouth=false,
		int global_z_min=100, int global_z_max=500) : is_07(is_07), 
													  is_separate_hands(is_separate_hands),
                                                      is_separate_face(is_separate_face),
                                                      is_earneck(is_earneck),
                                                      draw_face_pupils(draw_face_pupils),
                                                      draw_face_contour(draw_face_contour),
                                                      draw_face_inner_mouth(draw_face_inner_mouth),
                                                      global_z_min(global_z_min),
                                                      global_z_max(global_z_max) {

        pose_edges_num = -1;
        const index_pair *pose_arr_ptr;
        if (!is_07) {
            for (int i = 1; i < 4; i++) {
                _shift3[i] += 6 * 3;
                _shift4[i] += 6 * 4;
            }
            pose_edges_num = pose_edges_num_12;
            pose_arr_ptr = &(pose_edge_list_12[0]);
            if (is_earneck) {
                pose_edges_num = pose_edges_num_12_earneck;
                pose_arr_ptr = &(pose_edge_list_12_earneck[0]);
            }

        } else {
            pose_edges_num = pose_edges_num_07;
            pose_arr_ptr = &(pose_edge_list_07[0]);
            if (is_earneck) {
                pose_edges_num = pose_edges_num_07_earneck;
                pose_arr_ptr = &(pose_edge_list_07_earneck[0]);
            }
        }

        for (int i = 0; i < pose_edges_num; i++) {
            std::vector<int> edge_pair;
            for (int j = 0; j < 2; j++) {
                edge_pair.push_back(pose_arr_ptr[i][j]);
            }
            pose_edge_list.push_back(edge_pair);
        }

        face_edges_num = face_inner_edges_num;
        const index_pair * face_inner_arr_ptr = &(face_inner_edge_list[0]);

        for (int i = 0; i < face_inner_edges_num; i++)
        {
            std::vector<int> edge_pair;
            for (int j = 0; j < 2; j++) {
                edge_pair.push_back(face_inner_arr_ptr[i][j]);
            }
        	face_edge_list.push_back(edge_pair);
        }

        if (draw_face_pupils)
        {
        	face_edges_num += face_pupils_edges_num;
	        const index_pair * face_pupils_arr_ptr = &(face_pupils_edge_list[0]);

	        for (int i = 0; i < face_pupils_edges_num; i++)
	        {
	            std::vector<int> edge_pair;
	            for (int j = 0; j < 2; j++) {
	                edge_pair.push_back(face_pupils_arr_ptr[i][j]);
	            }
	        	face_edge_list.push_back(edge_pair);
	        }
        }
        if (draw_face_contour)
        {
        	face_edges_num += face_contour_edges_num;
	        const index_pair * face_contour_arr_ptr = &(face_contour_edge_list[0]);

	        for (int i = 0; i < face_contour_edges_num; i++)
	        {
	            std::vector<int> edge_pair;
	            for (int j = 0; j < 2; j++) {
	                edge_pair.push_back(face_contour_arr_ptr[i][j]);
	            }
	        	face_edge_list.push_back(edge_pair);
	        }
        }
        if (draw_face_inner_mouth)
        {
        	face_edges_num += face_inner_mouth_edges_num;
	        const index_pair * face_inner_mouth_arr_ptr = &(face_inner_mouth_edge_list[0]);

	        for (int i = 0; i < face_inner_mouth_edges_num; i++)
	        {
	            std::vector<int> edge_pair;
	            for (int j = 0; j < 2; j++) {
	                edge_pair.push_back(face_inner_mouth_arr_ptr[i][j]);
	            }
	        	face_edge_list.push_back(edge_pair);
	        }
        }

        if (is_separate_hands)
		{
        	hand_map_num = hand_edges_num;
		} else {
        	hand_map_num = 1;
        }

        if (is_separate_face) {
            face_map_num = face_edges_num;
        } else {
            face_map_num = 1;
        }
    }

    ~StickmanData_C() {}

    int size() {
        return _data.size();
    }

    int extracted_size() {
        return _extracted_data.size();
    }

    float get(int i) {
        return _data[i];
    }

    float get_extracted(int i) {
        return _extracted_data[i];
    }

    float *data() {
        return _data.data();
    }


    bool valid(int i) {
        return _valid[i];
    }

    float calcFaceMean() {
        float res = 0.0f;
        float cnt = 0.0;
        for (int i = 0; i < 70; ++i) {
            if (_extracted_data[i * 3 + _shift3[1]] > 0) {
                res += _extracted_data[i * 3 + _shift3[1] + 2];
                cnt += 1.0;
            }
        }
        res /= cnt;
        _faceMean = res;
        return res;
    }

    void print() {

        std::cout << size() << "\n";


        std::cout << "_data_size " << size() << "\n";
        std::cout << "_data" << "\n";
        for (int i = 0; i + 3 < size(); i += 4)
            std::cout << _data[i] << "\t" << _data[i + 1] << "\t" << _data[i + 2] << "\t" << _data[i + 3] << "\n";

        std::cout << "_extracted_data_size " << extracted_size() << "\n";
        std::cout << "_extracted_data" << "\n";
        for (int i = 0; i + 2 < extracted_size(); i += 3)
            std::cout << _extracted_data[i] << "\t" << _extracted_data[i + 1] << "\t" << _extracted_data[i + 2] << "\n";
    }

    void add_hand(int j, float thre, int *edges_num, float *input) {
//		thre=kp_threshold;//0.0f;
        for (int i_f = 0; i_f < edges_num[j] / 4; ++i_f) //if - number of a finger
        {
            bool add_finger = true;
            for (int i = 4 * i_f; i < 4 * i_f + 4; i++) {
                int idx1 = hand_edge_list[i][0];
                int idx2 = hand_edge_list[i][1];
                float conf1 = input[idx1 * 4 + 3 + _shift4[j]];
                float conf2 = input[idx2 * 4 + 3 + _shift4[j]];
                if ((conf1 < thre) || (conf2 < thre)) {
                    add_finger = false;
                }
            }
            if (!add_finger) {
                continue;
            }
            for (int i = 4 * i_f; i < 4 * i_f + 4; i++) {
                int idx1 = hand_edge_list[i][0];
                int idx2 = hand_edge_list[i][1];
                _valid[i + _shift4[j] / 4] = true;
                _extracted_data[idx1 * 3 + _shift3[j]] = input[idx1 * 4 +
                                                               _shift4[j]];                    //insert values
                _extracted_data[idx1 * 3 + _shift3[j] + 1] = input[idx1 * 4 + _shift4[j] + 1];
                _extracted_data[idx1 * 3 + _shift3[j] + 2] = input[idx1 * 4 + _shift4[j] + 2];
                _extracted_data[idx2 * 3 + _shift3[j]] = input[idx2 * 4 +
                                                               _shift4[j]];                    //insert values
                _extracted_data[idx2 * 3 + _shift3[j] + 1] = input[idx2 * 4 + _shift4[j] + 1];
                _extracted_data[idx2 * 3 + _shift3[j] + 2] = input[idx2 * 4 + _shift4[j] + 2];
            }
        }
    }

    void extract_valid_keypoints(float kp_threshold, float *input, int sz) {
        _extracted_data.clear();
        _valid.clear();
        _extracted_data.resize(3 * (sz / 4));
        _valid.resize(pose_edges_num + face_edges_num + hand_edges_num + hand_edges_num);
        fill(_valid.begin(), _valid.end(), false);
        fill(_extracted_data.begin(), _extracted_data.end(), 0.0f);

        float thre = kp_threshold;//0.0f;
        int edges_num[4] = {pose_edges_num, face_edges_num, hand_edges_num, hand_edges_num};
        int j = 0;
        for (int i = 0; i < edges_num[j]; ++i) {
            int idx1 = pose_edge_list[i][0];
            int idx2 = pose_edge_list[i][1];
            float conf1 = input[idx1 * 4 + 3 + _shift4[j]];
            float conf2 = input[idx2 * 4 + 3 + _shift4[j]];
            if ((conf1 > thre) && (conf2 > thre)) {
                _valid[i + _shift4[j] / 4] = true;
                _extracted_data[idx1 * 3 + _shift3[j]] = input[idx1 * 4 +
                                                               _shift4[j]];                    //insert values
                _extracted_data[idx1 * 3 + _shift3[j] + 1] = input[idx1 * 4 + _shift4[j] + 1];
                _extracted_data[idx1 * 3 + _shift3[j] + 2] = input[idx1 * 4 + _shift4[j] + 2];
                _extracted_data[idx2 * 3 + _shift3[j]] = input[idx2 * 4 +
                                                               _shift4[j]];                    //insert values
                _extracted_data[idx2 * 3 + _shift3[j] + 1] = input[idx2 * 4 + _shift4[j] + 1];
                _extracted_data[idx2 * 3 + _shift3[j] + 2] = input[idx2 * 4 + _shift4[j] + 2];
            }
        }

        j = 1;
        thre = 0.0;
        for (int i = 0; i < edges_num[j]; ++i) {
            //std::cout<<i<<"\n";
            int idx1 = face_edge_list[i][0];
            int idx2 = face_edge_list[i][1];
            float conf1 = input[idx1 * 4 + 3 + _shift4[j]];
            float conf2 = input[idx2 * 4 + 3 + _shift4[j]];
            if ((conf1 > thre) && (conf2 > thre)) {
                _valid[i + _shift4[j] / 4] = true;
                _extracted_data[idx1 * 3 + _shift3[j]] = input[idx1 * 4 +
                                                               _shift4[j]];                    //insert values
                _extracted_data[idx1 * 3 + _shift3[j] + 1] = input[idx1 * 4 + _shift4[j] + 1];
                _extracted_data[idx1 * 3 + _shift3[j] + 2] = input[idx1 * 4 + _shift4[j] + 2];
                _extracted_data[idx2 * 3 + _shift3[j]] = input[idx2 * 4 +
                                                               _shift4[j]];                    //insert values
                _extracted_data[idx2 * 3 + _shift3[j] + 1] = input[idx2 * 4 + _shift4[j] + 1];
                _extracted_data[idx2 * 3 + _shift3[j] + 2] = input[idx2 * 4 + _shift4[j] + 2];
            }
        }

        j = 2;
        add_hand(j, kp_threshold, edges_num, input);
        j = 3;
        add_hand(j, kp_threshold, edges_num, input);
    }

    bool getLineCoords(int i, LineT &result) {

        if (i < pose_edges_num) {
            result.x1 = _extracted_data[3 * pose_edge_list[i][0] + _shift3[0]];
            result.y1 = _extracted_data[3 * pose_edge_list[i][0] + _shift3[0] + 1];
            result.x2 = _extracted_data[3 * pose_edge_list[i][1] + _shift3[0]];
            result.y2 = _extracted_data[3 * pose_edge_list[i][1] + _shift3[0] + 1];
            result.z1 = _extracted_data[3 * pose_edge_list[i][0] + _shift3[0] + 2];
            result.z2 = _extracted_data[3 * pose_edge_list[i][1] + _shift3[0] + 2];
            return (result.x1 > 0 && result.x2 > 0);
        }
        i -= pose_edges_num;
        if (i < face_edges_num) {
            result.x1 = _extracted_data[3 * face_edge_list[i][0] + _shift3[1]];
            result.y1 = _extracted_data[3 * face_edge_list[i][0] + _shift3[1] + 1];
            result.x2 = _extracted_data[3 * face_edge_list[i][1] + _shift3[1]];
            result.y2 = _extracted_data[3 * face_edge_list[i][1] + _shift3[1] + 1];
            result.z1 = _extracted_data[3 * face_edge_list[i][0] + _shift3[1] + 2];
            result.z2 = _extracted_data[3 * face_edge_list[i][1] + _shift3[1] + 2];
            return (result.x1 > 0 && result.x2 > 0);
        }
        i -= face_edges_num;
        if (i < hand_edges_num) {
            result.x1 = _extracted_data[3 * hand_edge_list[i][0] + _shift3[2]];
            result.y1 = _extracted_data[3 * hand_edge_list[i][0] + _shift3[2] + 1];
            result.x2 = _extracted_data[3 * hand_edge_list[i][1] + _shift3[2]];
            result.y2 = _extracted_data[3 * hand_edge_list[i][1] + _shift3[2] + 1];
            result.z1 = _extracted_data[3 * hand_edge_list[i][0] + _shift3[2] + 2];
            result.z2 = _extracted_data[3 * hand_edge_list[i][1] + _shift3[2] + 2];
            return (result.x1 > 0 && result.x2 > 0);
        }
        i -= hand_edges_num;
        if (i < hand_edges_num) {
            result.x1 = _extracted_data[3 * hand_edge_list[i][0] + _shift3[3]];
            result.y1 = _extracted_data[3 * hand_edge_list[i][0] + _shift3[3] + 1];
            result.x2 = _extracted_data[3 * hand_edge_list[i][1] + _shift3[3]];
            result.y2 = _extracted_data[3 * hand_edge_list[i][1] + _shift3[3] + 1];
            result.z1 = _extracted_data[3 * hand_edge_list[i][0] + _shift3[3] + 2];
            result.z2 = _extracted_data[3 * hand_edge_list[i][1] + _shift3[3] + 2];
            return (result.x1 > 0 && result.x2 > 0);
        } else
            return false;
    }


    float distance(float x0, float y0, float x1, float y1, float x, float y) {
        if ((x0 == x1) && (y0 == y1))
            return (sqrtf(powf(x0 - x, 2.0f) + powf(y0 - y, 2.0f)));
        float numenator, denomenator;
        numenator = (y0 - y1) * x + (x1 - x0) * y + (x0 * y1 - x1 * y0);
        denomenator = sqrtf(powf(x1 - x0, 2.0f) + powf(y1 - y0, 2.0f));
        return (fabsf(numenator / denomenator));
    }

    int clip(int x, int maxval) {
        return std::max(std::min(x, maxval), 0);
    }

    int mmin(int x1, int x2, int bw) {
        return min(x1, x2) - bw - 1;
    }

    int mmax(int x1, int x2, int bw) {
        return max(x1, x2) + bw + 1;
    }

    void drawLine(const LineT &line, int channel, float bw, int w, int h, float val, float *output) {

        int x1i = (int) floor(line.x1 + 0.5);
        int y1i = (int) floor(line.y1 + 0.5);
        int x2i = (int) floor(line.x2 + 0.5);
        int y2i = (int) floor(line.y2 + 0.5);

        float n[3];
        n[0] = y1i - y2i;
        n[1] = x2i - x1i;
        n[2] = (x1i * y2i - x2i * y1i);
        float s = sqrtf(n[0] * n[0] + n[1] * n[1]);

        float deps = 1e-5;
        n[0] /= (s + deps);
        n[1] /= (s + deps);
        n[2] /= (s + deps);

        int bwi = (int) (bw + 1.0);
        int fromi = clip(mmin(x1i, x2i, bwi), w);
        int toi = clip(mmax(x1i, x2i, bwi), w);

        int fromj = clip(mmin(y1i, y2i, bwi), h);
        int toj = clip(mmax(y1i, y2i, bwi), h);

//        std::cout << fromi << " " << toi << " " << fromj << " " << toj << " " << std::endl;
        for (int j = fromj; j < toj; j++) {
            for (int i = fromi; i < toi; i++) {
                float signed_dist = i * n[0] + j * n[1] + n[2];
                float dist_val = fabs(signed_dist);
                float p_x = i - n[0] * signed_dist;
                float p_y = j - n[1] * signed_dist;

                if (dist_val >= bw + 1.0) {
                    continue;
                }
                if ((p_x - line.x1) * (p_x - line.x2) + (p_y - line.y1) * (p_y - line.y2) > 0) {
                    float d1 = sqrt((i - line.x1) * (i - line.x1) + (j - line.y1) * (j - line.y1));
                    float d2 = sqrt((i - line.x2) * (i - line.x2) + (j - line.y2) * (j - line.y2));
                    float ep_dist_val = std::min(d1, d2);
                    if (ep_dist_val >= bw + 1.0) {
                        continue;
                    }
                    dist_val = ep_dist_val;
                }

                int addr = i + j * w + channel * w * h;
                if (dist_val <= bw) {
                    output[addr] = val;
                } else {
                    if (dist_val < bw + 1.0) {
                        float gamma = bw + 1.0 - dist_val;
                        output[addr] = gamma * val + (1 - gamma) * output[addr];
                    }
                }

            }
        }
    }

    void drawPose2(float *output, int w, int h, float bw = 3) {
        LineT line;
        for (int edge = 0; edge < pose_edges_num; ++edge) {
            if (getLineCoords(edge, line)) {
                float mean = (line.z1 + line.z2) / 2.0f;
                drawLine(line, edge, bw, w, h, mean, output);
            }
        }
    }

    void drawPose(float *output, int w, int h, int bw = 3) {
        LineT line;
        for (int edge = 0; edge < pose_edges_num; ++edge) {
            if (getLineCoords(edge, line)) {

                int fromi = min(line.x1, line.x2);
                int toi = max(line.x1, line.x2);

                int fromj = min(line.y1, line.y2);
                int toj = max(line.y1, line.y2);
                float mean = (line.z1 + line.z2) / 2.0f;

                for (int j = fromj; j <= toj; ++j) {
                    for (int i = fromi; i <= toi; ++i) {
#pragma unroll
                        for (int jp = -bw; jp <= bw; ++jp) {
#pragma unroll
                            for (int ip = -bw; ip <= bw; ++ip) {
                                if ((j + jp < 0) || (j + jp >= h) || (i + ip < 0) || (i + ip >= w))
                                    continue;
                                else {
                                    if (distance(line.x1, line.y1, line.x2, line.y2, i + ip, j + jp) <= bw) {
                                        output[(i + ip + (j + jp) * w) + edge * w * h] = mean;//color[0];
                                        //output[(i+ip+(j+jp)*w)+1+edge*w*h]=mean;//color[1];
                                        //output[(i+ip+(j+jp)*w)+2+edge*w*h]=mean;//color[2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void drawFace2(float *output, int w, int h, float bw = 3) {

        calcFaceMean();
        LineT line;
        for (int edge = 0; edge < face_edges_num; ++edge) {
            if (getLineCoords(edge + pose_edges_num, line)) {
//                std::cout << pose_edges_num+2*hand_map_num << std::endl;
                if (is_separate_face) {
                    drawLine(line, pose_edges_num + 2 * hand_map_num + edge, bw, w, h, _faceMean, output);
                } else {
                    drawLine(line, pose_edges_num + 2 * hand_map_num, bw, w, h, _faceMean, output);
                }
            }
        }
    }

    void drawFace(float *output, int w, int h, int bw = 3) {

        calcFaceMean();
        LineT line;
        for (int edge = 0; edge < face_edges_num; ++edge) {
            if (getLineCoords(edge + pose_edges_num, line)) {
                int fromi = min(line.x1, line.x2);
                int toi = max(line.x1, line.x2);

                int fromj = min(line.y1, line.y2);
                int toj = max(line.y1, line.y2);
                for (int j = fromj; j <= toj; ++j) {
                    for (int i = fromi; i <= toi; ++i) {
#pragma unroll
                        for (int jp = -bw; jp <= bw; ++jp) {
#pragma unroll
                            for (int ip = -bw; ip <= bw; ++ip) {
                                if ((j + jp < 0) || (j + jp >= h) || (i + ip < 0) || (i + ip >= w))
                                    continue;
                                else {
                                    if (distance(line.x1, line.y1, line.x2, line.y2, i + ip, j + jp) <= bw) {
                                        output[(i + ip + (j + jp) * w) +
                                               (pose_edges_num + 2) * w * h] = _faceMean;//color[0];
                                        //output[(i+ip+(j+jp)*w)+1+(pose_edges_num+2)*w*h]=_faceMean;//color[1];
//										output[(i+ip+(j+jp)*w)+2+(pose_edges_num+2)*w*h]=_faceMean;//color[2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    void drawHands(float *output, int w, int h, int bw = 1) {
        LineT line;
#pragma unroll
        for (int ha = 0; ha < 2; ++ha) {
            for (int edge = 0; edge < hand_edges_num; ++edge) {
                int l = edge + pose_edges_num + face_edges_num;
                if (ha > 0)
                    l += hand_edges_num;
                if (getLineCoords(l, line)) {
                    int fromi = min(line.x1, line.x2);
                    int toi = max(line.x1, line.x2);

                    int fromj = min(line.y1, line.y2);
                    int toj = max(line.y1, line.y2);
                    float mean = (line.z1 + line.z2) / 2.0f;

                    for (int j = fromj; j <= toj; ++j) {
                        for (int i = fromi; i <= toi; ++i) {
#pragma unroll
                            for (int jp = -bw; jp <= bw; ++jp) {
#pragma unroll
                                for (int ip = -bw; ip <= bw; ++ip) {
                                    if ((j + jp < 0) || (j + jp >= h) || (i + ip < 0) || (i + ip >= w))
                                        continue;
                                    else {
                                        if (distance(line.x1, line.y1, line.x2, line.y2, i + ip, j + jp) <= bw) {
                                            output[(i + ip + (j + jp) * w) +
                                                   (pose_edges_num + ha) * w * h] = mean;//color[0];
//											output[(i+ip+(j+jp)*w)+1+(pose_edges_num+ha)*w*h]=mean;//color[1];
//											output[(i+ip+(j+jp)*w)+2+(pose_edges_num+ha)*w*h]=mean;//color[2];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void drawHands2(float *output, int w, int h, float bw = 1) {
        LineT line;
#pragma unroll
        for (int ha = 0; ha < 2; ++ha) {
            for (int edge = 0; edge < hand_edges_num; ++edge) {
                int l = edge + pose_edges_num + face_edges_num;
                if (ha > 0)
                    l += hand_edges_num;
                if (getLineCoords(l, line)) {
                    float mean = (line.z1 + line.z2) / 2.0f;
                    if (is_separate_hands) {
                        drawLine(line, pose_edges_num + ha * hand_edges_num + edge, bw, w, h, mean, output);
                    } else {
                        drawLine(line, pose_edges_num + ha, bw, w, h, mean, output);
                    }
                }
            }
        }
    }


    void drawStickman(float *output, int sz1, float *input, int sz2, int w, int h, int lineWidthPose = 3,
                      int lineWidthFaceHand = 3) {
        int minDepth = input[2];
        int maxDepth = input[2];

        _data.resize(w * h);
        for (int i = 0; i + 2 < sz2; i += 4) {
            _data[i] = input[i];
            _data[i + 1] = input[i + 1];
            _data[i + 2] = input[i + 2];
            _data[i + 3] = input[i + 3];
//			if (minDepth>input[i+2])
//				minDepth=input[i+2];
//			if (maxDepth<input[i+2])
//				maxDepth=input[i+2];
        }

//		float norm=maxDepth=0.0f;
//		if (fabsf(maxDepth-minDepth)<1e-6)
//			norm=maxDepth;
//		else
//			norm=maxDepth-minDepth;
        for (int i = 0; i + 2 < sz2; i += 4) {
            float temp = std::min(std::max((_data[i + 2] - global_z_min) / (global_z_max - global_z_min), (float) 0.0),
                                  (float) 1.0) * 255;
            _data[i + 2] = temp;
        }

        extract_valid_keypoints(0.05f, data(), sz2);

        drawPose(output, w, h, lineWidthPose);
        drawFace(output, w, h, lineWidthFaceHand);
        drawHands(output, w, h, lineWidthFaceHand);

    }

    void drawStickman2(float *output, int sz1, float *input, int sz2, int w, int h, float lineWidthPose = 3,
                       float lineWidthFaceHand = 3) {
        _data.resize(sz2);
        for (int i = 0; i + 2 < sz2; i += 4) {
            _data[i] = input[i];
            _data[i + 1] = input[i + 1];
            _data[i + 2] = input[i + 2];
            _data[i + 3] = input[i + 3];
        }

        for (int i = 0; i + 2 < sz2; i += 4) {
            float temp = std::min(std::max((_data[i + 2] - global_z_min) / (global_z_max - global_z_min), (float) 0.0),
                                  (float) 1.0) * 255;
            _data[i + 2] = temp;
        }

        extract_valid_keypoints(0.05f, data(), sz2);

//        std::cout << " output size " << sz1 << std::endl;

//        std::cout << "extracted keypoints " << std::endl;

        drawPose2(output, w, h, lineWidthPose);

//        std::cout << "drawn pose  " << std::endl;

        drawFace2(output, w, h, lineWidthFaceHand);

//        std::cout << "drawn face " << std::endl;
        drawHands2(output, w, h, lineWidthFaceHand);

//        std::cout << "drawn hands " << std::endl;

    }

};
