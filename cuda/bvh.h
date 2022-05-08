#ifndef BVH_H
#define BVH_H

#include <algorithm>
#include <thrust/sort.h>

#include "sceneobject.h"
#include "sceneobject_list.h"

class bvh_node : public sceneobject {
    public:
        sceneobject* left;
        sceneobject* right;
        bounding_box box;

        __device__ bvh_node();

        __device__ bvh_node(sceneobject_list& scene_list, curandState *local_rand_state) : bvh_node(scene_list.list, 0, scene_list.list_size, local_rand_state) {}

        __device__ bvh_node(sceneobject **src_objects, size_t start, size_t end, curandState *local_rand_state);

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
        __device__ virtual bool get_bounding_box(bounding_box& output_box) const override;
};

__device__ bool box_compare(const sceneobject* a, const sceneobject* b, int axis) {
    bounding_box box_a;
    bounding_box box_b;
    a->get_bounding_box(box_a);
    b->get_bounding_box(box_b);
    //if (!a->get_bounding_box(box_a) || !b->get_bounding_box(box_b))
    //    printf("No bounding box in bvh_node constructor.\n");

    return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ void bvh_sort(sceneobject** object_list, int start, int end, int axis) {

    int i, j;
    sceneobject* key;
    for(i = start + 1; i < end; i++) {
        key = object_list[i];
        j = i - 1;

        while (j >= start && box_compare(key, object_list[j], axis)) {
            object_list[j+1] = object_list[j];
            j = j - 1;

        }
        object_list[j+1] = key;
    }

}

__device__ bvh_node::bvh_node(sceneobject** src_objects, size_t start, size_t end, curandState *local_rand_state) {
    auto objects = src_objects;
    size_t range = end - start;

    // Random int between 0 and 2 inclusive
    int axis = int(curand_uniform(local_rand_state) * (2.0f - 0.0f) + 0.0f);

    if (range == 1) {
        left = right = objects[start];
    } else if (range == 2) {
        if (box_compare(objects[start], objects[start+1], axis)) {
            left = objects[start];
            right = objects[start+1];
        } else {
            left = objects[start+1];
            right = objects[start];
        }
    } else {

        bvh_sort(objects, start, end, axis);

        size_t mid = start + range/2;
        left = new bvh_node(objects, start, mid, local_rand_state);
        right = new bvh_node(objects, mid, end, local_rand_state);
    }

    bounding_box box_left, box_right;
    left->get_bounding_box(box_left);
    right->get_bounding_box(box_right);

    //if (!left->get_bounding_box(box_left) || !right->get_bounding_box(box_right))
    //    printf("error");

    box = surrounding_box(box_left, box_right);

}


__device__ bool bvh_node::get_bounding_box(bounding_box &output_box) const {
    output_box = box;
    return true;
}

__device__ bool bvh_node::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
    if (!box.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

#endif
