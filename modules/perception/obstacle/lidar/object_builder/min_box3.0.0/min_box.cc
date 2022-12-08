/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/perception/obstacle/lidar/object_builder/min_box/min_box.h"

#include <limits>

#include "modules/perception/common/convex_hullxy.h"
#include "modules/perception/common/geometry_util.h"
#include "modules/perception/common/pcl_types.h"

namespace apollo
{
namespace perception
{

using pcl_util::PointCloud;
using pcl_util::PointCloudPtr;

const float EPSILON = 1e-6;

bool MinBoxObjectBuilder::Build(const ObjectBuilderOptions &options,
                                std::vector<std::shared_ptr<Object>> *objects)
{
  if (objects == nullptr) { return false; }

  // traverse all objects
  for (size_t i = 0; i < objects->size(); ++i)
  {
    if ((*objects)[i])
    {
      (*objects)[i]->id = i;
      BuildObject(options, (*objects)[i]);
    }
  }

  return true;
}

double MinBoxObjectBuilder::ComputeAreaAlongOneEdge(
    std::shared_ptr<Object> obj, size_t first_in_point, Eigen::Vector3d *center,
    double *lenth, double *width, Eigen::Vector3d *dir)
{
  // Here are point A, B, O, N. N is at line AB and ON is perpendicular to AB.
  // AB will be the largest side and O will be one of other vertices.
  std::vector<Eigen::Vector3d> ns;   // save all N for every O
  Eigen::Vector3d v(0.0, 0.0, 0.0);  // save O that is most further from AB
  Eigen::Vector3d vn(0.0, 0.0, 0.0); // save corresponding N
  Eigen::Vector3d n(0.0, 0.0, 0.0);  // this is point N

  double len = 0; // max distance between vertices
  double wid = 0; // max distance between vertices and this side(I think this side is the largest side)

  // calculate wid
  size_t index = (first_in_point + 1) % obj->polygon.points.size();
  for (size_t i = 0; i < obj->polygon.points.size(); ++i)
  {
    if (i != first_in_point && i != index)
    {
      // compute v
      Eigen::Vector3d o(0.0, 0.0, 0.0); // current vertice
      Eigen::Vector3d a(0.0, 0.0, 0.0); // next vertice of input
      Eigen::Vector3d b(0.0, 0.0, 0.0); // input vertice
      o[0] = obj->polygon.points[i].x;
      o[1] = obj->polygon.points[i].y;
      o[2] = 0;
      b[0] = obj->polygon.points[first_in_point].x;
      b[1] = obj->polygon.points[first_in_point].y;
      b[2] = 0;
      a[0] = obj->polygon.points[index].x;
      a[1] = obj->polygon.points[index].y;
      a[2] = 0;

      double k = ((a[0] - o[0]) * (b[0] - a[0]) + (a[1] - o[1]) * (b[1] - a[1]));     // OA*AB
      k        = k / ((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])); // OA*AB/|AB|
      k        = k * -1;
      // n is pedal of src;
      n[0] = (b[0] - a[0]) * k + a[0];
      n[1] = (b[1] - a[1]) * k + a[1];
      n[2] = 0;
      // compute height from src to line
      Eigen::Vector3d edge1 = o - b; // vector from b to o
      Eigen::Vector3d edge2 = a - b; // vector from a to o
      // cross product
      double height = fabs(edge1[0] * edge2[1] - edge2[0] * edge1[1]);
      height        = height / sqrt(edge2[0] * edge2[0] + edge2[1] * edge2[1]);
      if (height > wid)
      {
        wid = height;
        v   = o;
        vn  = n;
      }
    }
    else
    {
      n[0] = obj->polygon.points[i].x;
      n[1] = obj->polygon.points[i].y;
      n[2] = 0;
    }
    ns.push_back(n);
  } // endfor: have calculated wid

  // calculate len
  size_t point_num1 = 0;
  size_t point_num2 = 0;
  for (size_t i = 0; i < ns.size() - 1; ++i)
  {
    Eigen::Vector3d p1 = ns[i];
    for (size_t j = i + 1; j < ns.size(); ++j)
    {
      Eigen::Vector3d p2 = ns[j];
      double dist        = sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]));
      if (dist > len)
      {
        len        = dist;
        point_num1 = i;
        point_num2 = j;
      }
    }
  } // endfor: have calculated len

  Eigen::Vector3d vp1 = v + ns[point_num1] - vn;
  Eigen::Vector3d vp2 = v + ns[point_num2] - vn;
  // now we have calculated four vertices of min box: ns[point_num1], ns[point_num2], vp1, vp2
  (*center)    = (vp1 + vp2 + ns[point_num1] + ns[point_num2]) / 4;
  (*center)[2] = obj->polygon.points[0].z;
  if (len > wid)
  {
    *dir = ns[point_num2] - ns[point_num1]; // direction of the largest side
  }
  else
  {
    *dir = vp1 - ns[point_num1]; // vertical direction of the largest side
  }
  *lenth = len > wid ? len : wid;
  *width = len > wid ? wid : len;
  return (*lenth) * (*width);
}

void MinBoxObjectBuilder::ReconstructPolygon(const Eigen::Vector3d &ref_ct,
                                             std::shared_ptr<Object> obj)
{
  if (obj->polygon.points.size() <= 0)
  {
    return;
  }
  size_t max_point_index = 0;
  size_t min_point_index = 0;
  Eigen::Vector3d p;
  p[0] = obj->polygon.points[0].x;
  p[1] = obj->polygon.points[0].y;
  p[2] = obj->polygon.points[0].z;

  Eigen::Vector3d max_point = p - ref_ct; // vector from reference center to vertice
  Eigen::Vector3d min_point = p - ref_ct;
  // traverse all vertices of polygon to find two vertices to consist a vector,
  // left of which all other vertices locate at.
  for (size_t i = 1; i < obj->polygon.points.size(); ++i)
  {
    Eigen::Vector3d p;
    p[0] = obj->polygon.points[i].x;
    p[1] = obj->polygon.points[i].y;
    p[2] = obj->polygon.points[i].z;

    Eigen::Vector3d ray = p - ref_ct;
    // clock direction; right of vector
    if (max_point[0] * ray[1] - ray[0] * max_point[1] < EPSILON)
    {
      max_point       = ray;
      max_point_index = i;
    }
    // unclock direction; left of vector
    if (min_point[0] * ray[1] - ray[0] * min_point[1] > EPSILON)
    {
      min_point       = ray;
      min_point_index = i;
    }
  }
  Eigen::Vector3d line = max_point - min_point; // all points should locate at left of this vector
  double total_len     = 0;                     // The perimeter of the convex polygon
  double max_dis       = 0;                     // largest side of the convex polygon
  bool has_out         = false;                 // points that locate at right of this vector exist.
  for (size_t i = min_point_index, count = 0;
       count < obj->polygon.points.size();
       i = (i + 1) % obj->polygon.points.size(), ++count)
  {
    Eigen::Vector3d p_x;
    p_x[0]   = obj->polygon.points[i].x;
    p_x[1]   = obj->polygon.points[i].y;
    p_x[2]   = obj->polygon.points[i].z;
    size_t j = (i + 1) % obj->polygon.points.size();
    // j:non-end
    if (j != min_point_index && j != max_point_index)
    {
      Eigen::Vector3d p;
      p[0] = obj->polygon.points[j].x;
      p[1] = obj->polygon.points[j].y;
      p[2] = obj->polygon.points[j].z;

      Eigen::Vector3d ray = p - min_point;
      if (line[0] * ray[1] - ray[0] * line[1] < EPSILON)
      {
        double dist = sqrt((p[0] - p_x[0]) * (p[0] - p_x[0]) + (p[1] - p_x[1]) * (p[1] - p_x[1]));
        total_len += dist;
        if (dist - max_dis > EPSILON)
        {
          max_dis = dist;
        }
      }
      else
      {
        // outline
        has_out = true;
      }
    }
    // j:end; i:end
    else if ((i == min_point_index && j == max_point_index) || (i == max_point_index && j == min_point_index))
    {
      size_t k = (j + 1) % obj->polygon.points.size();
      Eigen::Vector3d p_k;
      p_k[0] = obj->polygon.points[k].x;
      p_k[1] = obj->polygon.points[k].y;
      p_k[2] = obj->polygon.points[k].z;
      Eigen::Vector3d p_j;
      p_j[0] = obj->polygon.points[j].x;
      p_j[1] = obj->polygon.points[j].y;
      p_j[2] = obj->polygon.points[j].z;

      Eigen::Vector3d ray = p - min_point;
      if (line[0] * ray[1] - ray[0] * line[1] < 0)
      {
      }
      else
      {
        // outline
        has_out = true;
      }
    }
    // j:end; i:non-end
    else if (j == min_point_index || j == max_point_index)
    {
      Eigen::Vector3d p;
      p[0] = obj->polygon.points[j].x;
      p[1] = obj->polygon.points[j].y;
      p[2] = obj->polygon.points[j].z;

      Eigen::Vector3d ray = p_x - min_point;
      if (line[0] * ray[1] - ray[0] * line[1] < EPSILON)
      {
        double dist = sqrt((p[0] - p_x[0]) * (p[0] - p_x[0]) + (p[1] - p_x[1]) * (p[1] - p_x[1]));
        total_len += dist;
        if (dist > max_dis) { max_dis = dist; }
      }
      else
      {
        // outline
        has_out = true;
      }
    }
  } // endfor: have find largest side and whether outline points exist

  size_t count    = 0;
  double min_area = std::numeric_limits<double>::max();
  for (size_t i = min_point_index;
       count < obj->polygon.points.size();
       i = (i + 1) % obj->polygon.points.size(), ++count)
  {
    Eigen::Vector3d p_x;
    p_x[0] = obj->polygon.points[i].x;
    p_x[1] = obj->polygon.points[i].y;
    p_x[2] = obj->polygon.points[i].z;

    size_t j = (i + 1) % obj->polygon.points.size();
    Eigen::Vector3d p_j;
    p_j[0] = obj->polygon.points[j].x;
    p_j[1] = obj->polygon.points[j].y;
    p_j[2] = obj->polygon.points[j].z;

    double dist = sqrt((p_x[0] - p_j[0]) * (p_x[0] - p_j[0]) + (p_x[1] - p_j[1]) * (p_x[1] - p_j[1]));
    if (dist < max_dis && (dist / total_len) < 0.5) // find the largest side
    {
      continue;
    }
    if (j != min_point_index && j != max_point_index)
    {
      Eigen::Vector3d p;
      p[0] = obj->polygon.points[j].x;
      p[1] = obj->polygon.points[j].y;
      p[2] = obj->polygon.points[j].z;

      Eigen::Vector3d ray = p - min_point;
      if (line[0] * ray[1] - ray[0] * line[1] < 0)
      {
        Eigen::Vector3d center;
        double length = 0;
        double width  = 0;
        Eigen::Vector3d dir;
        double area = ComputeAreaAlongOneEdge(obj, i, &center, &length, &width, &dir);
        if (area < min_area)
        {
          obj->center    = center;
          obj->length    = length;
          obj->width     = width;
          obj->direction = dir;
          min_area       = area;
        }
      }
      else
      {
        // outline
      }
    }
    else if ((i == min_point_index && j == max_point_index) || (i == max_point_index && j == min_point_index))
    {
      if (!has_out)
      {
        continue;
      }
      Eigen::Vector3d center;
      double length = 0;
      double width  = 0;
      Eigen::Vector3d dir;
      double area = ComputeAreaAlongOneEdge(obj, i, &center, &length, &width, &dir);
      if (area < min_area)
      {
        obj->center    = center;
        obj->length    = length;
        obj->width     = width;
        obj->direction = dir;
        min_area       = area;
      }
    }
    else if (j == min_point_index || j == max_point_index)
    {
      Eigen::Vector3d p;
      p[0]                = obj->polygon.points[i].x;
      p[1]                = obj->polygon.points[i].y;
      p[2]                = obj->polygon.points[i].z;
      Eigen::Vector3d ray = p - min_point;
      if (line[0] * ray[1] - ray[0] * line[1] < 0)
      {
        Eigen::Vector3d center;
        double length = 0.0;
        double width  = 0.0;
        Eigen::Vector3d dir;
        double area = ComputeAreaAlongOneEdge(obj, i, &center, &length, &width, &dir);
        if (area < min_area)
        {
          obj->center    = center;
          obj->length    = length;
          obj->width     = width;
          obj->direction = dir;
          min_area       = area;
        }
      }
      else
      {
        // outline
      }
    }
  } // endfor: have found the largest side and gotten min box
  obj->direction.normalize();
}

// use PCL to compute points that consist convex polygon
void MinBoxObjectBuilder::ComputePolygon2dxy(std::shared_ptr<Object> obj)
{
  Eigen::Vector4f min_pt;
  Eigen::Vector4f max_pt;
  pcl_util::PointCloudPtr cloud = obj->cloud;
  SetDefaultValue(cloud, obj, &min_pt, &max_pt);
  if (cloud->points.size() < 4u)
  {
    return;
  }
  GetCloudMinMax3D<pcl_util::Point>(cloud, &min_pt, &max_pt);
  obj->height          = static_cast<double>(max_pt[2]) - static_cast<double>(min_pt[2]);
  const double min_eps = 10 * std::numeric_limits<double>::epsilon();
  // double min_eps = 0.1;
  // if ((max_pt[0] - min_pt[0]) < min_eps) {
  //     cloud_->points[0].x += min_eps;
  // }
  // if ((max_pt[1] - min_pt[1]) < min_eps) {
  //     cloud_->points[0].y += min_eps;
  // }
  const double diff_x = cloud->points[1].x - cloud->points[0].x;
  const double diff_y = cloud->points[1].y - cloud->points[0].y;

  size_t idx = 0;
  // find first unclock point for this line
  for (idx = 2; idx < cloud->points.size(); ++idx)
  {
    const double tdiff_x = cloud->points[idx].x - cloud->points[0].x;
    const double tdiff_y = cloud->points[idx].y - cloud->points[0].y;
    if ((diff_x * tdiff_y - tdiff_x * diff_y) > min_eps) // unclock
    {
      break;
    }
  }
  if (idx >= cloud->points.size())
  {
    cloud->points[0].x += min_eps;
    cloud->points[0].y += min_eps;
    cloud->points[1].x -= min_eps;
  }

  PointCloudPtr pcd_xy(new PointCloud);
  for (size_t i = 0; i < cloud->points.size(); ++i) // 所有的点位于同一水平面
  {
    pcl_util::Point p = cloud->points[i];
    p.z               = min_pt[2];
    pcd_xy->push_back(p);
  }

  ConvexHull2DXY<pcl_util::Point> hull;
  hull.setInputCloud(pcd_xy);
  hull.setDimension(2);
  std::vector<pcl::Vertices> poly_vt;
  PointCloudPtr plane_hull(new PointCloud);
  hull.Reconstruct2dxy(plane_hull, &poly_vt);

  if (poly_vt.size() == 1u)
  {
    std::vector<int> ind(poly_vt[0].vertices.begin(),
                         poly_vt[0].vertices.end());
    TransformPointCloud(plane_hull, ind, &obj->polygon);
  }
  else
  {
    obj->polygon.points.resize(4);
    obj->polygon.points[0].x = static_cast<double>(min_pt[0]);
    obj->polygon.points[0].y = static_cast<double>(min_pt[1]);
    obj->polygon.points[0].z = static_cast<double>(min_pt[2]);

    obj->polygon.points[1].x = static_cast<double>(min_pt[0]);
    obj->polygon.points[1].y = static_cast<double>(max_pt[1]);
    obj->polygon.points[1].z = static_cast<double>(min_pt[2]);

    obj->polygon.points[2].x = static_cast<double>(max_pt[0]);
    obj->polygon.points[2].y = static_cast<double>(max_pt[1]);
    obj->polygon.points[2].z = static_cast<double>(min_pt[2]);

    obj->polygon.points[3].x = static_cast<double>(max_pt[0]);
    obj->polygon.points[3].y = static_cast<double>(min_pt[1]);
    obj->polygon.points[3].z = static_cast<double>(min_pt[2]);
  }
}

void MinBoxObjectBuilder::ComputeGeometricFeature(const Eigen::Vector3d &ref_ct,
                                                  std::shared_ptr<Object> obj)
{
  ComputePolygon2dxy(obj); // compute points that consist convex polygon
  ReconstructPolygon(ref_ct, obj);
}

void MinBoxObjectBuilder::BuildObject(ObjectBuilderOptions options,
                                      std::shared_ptr<Object> object)
{
  ComputeGeometricFeature(options.ref_center, object);
}

} // namespace perception
} // namespace apollo
