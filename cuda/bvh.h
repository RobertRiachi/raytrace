#ifndef BVH_H
#define BVH_H

#include "sceneobject.h"
#include "sceneobject_list.h"

class bvh_node : public sceneobject {
    public:
        sceneobject* left;
        sceneobject* right;
        bounding_box box;

        __device__ bvh_node();

        __device__ bvh_node(const sceneobject_list& scene_list) : bvh_node(scene_list.list, 0, scene_list.list.size()) {}

        __device__ bvh_node(const std::vector<hittable *>& src_objects, size_t start, size_t end);

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
        __device__ virtual bool get_bounding_box(bounding_box& output_box) const override;
};

bool bvh_node::get_bounding_box(bounding_box &output_box) const {
    output_box = box;
    return true;
}

bool bvh_node::hit(const int &r, float t_min, float t_max, hit_record &rec) const {
    if (!box.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

#endif
