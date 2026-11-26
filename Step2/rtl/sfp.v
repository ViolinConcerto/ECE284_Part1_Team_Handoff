// Step2 SFU: one signed output-channel accumulator.
module sfp #(
    parameter bw = 4,
    parameter psum_bw = 16
)(
    input clk,
    input reset,
    input clear,
    input acc,
    input relu,
    input signed [psum_bw-1:0] in,
    output signed [psum_bw-1:0] out
);
    reg signed [psum_bw-1:0] accumulator;

    always @(posedge clk) begin
        if (reset)
            accumulator <= 0;
        else if (clear)
            accumulator <= 0;
        else if (acc)
            accumulator <= accumulator + in;
        else if (relu && accumulator[psum_bw-1])
            accumulator <= 0;
        // Idle and nonnegative ReLU retain the registered sum.
    end

    assign out = accumulator;
endmodule
