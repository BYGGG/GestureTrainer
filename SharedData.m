classdef SharedData < handle
    properties
        project_dir
    end
    
    methods (Static)
        function obj = getInstance()
            persistent uniqueInstance
            if isempty(uniqueInstance)
                obj = SharedData();
                uniqueInstance = obj;
            else
                obj = uniqueInstance;
            end
        end
    end
end
