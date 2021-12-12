#ifndef SCENEOBJECT_LIST_H
#define SCENEOBJECT_LIST_H

#include <memory>
#include <vector>
#include "sceneobject.h"

using std::shared_ptr;
using std::make_shared;

class sceneobject_list: public sceneobject {
    public:

        std::vector<shared_ptr<sceneobject>> objects;

        sceneobject_list() {}
        sceneobject_list(shared_ptr<sceneobject> object) { add(object); }

        void clear() {objects.clear();}

        void add(shared_ptr<sceneobject> object) {objects.push_back(object);}

        virtual bool hit (const ray& r, double t_min, double t_max, hit_record& rec) const override;
};

bool sceneobject_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;

    for (const auto&object : objects) {
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif